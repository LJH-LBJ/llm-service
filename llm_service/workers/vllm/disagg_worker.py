# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import asyncio
import os
from typing import Any

import msgspec
import numpy as np
from numpy.typing import NDArray
import zmq
import zmq.asyncio

from llm_service.protocol.protocol import (
    FailureResponse,
    GenerationRequest,
    GenerationResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ProfileRequest,
    ProfileResponse,
    RequestType,
    ResponseType,
)
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


class DisaggWorker:
    def __init__(
        self,
        engine: EngineClient,
        address: str,
        proxy_addr: str,
    ):
        self.engine = engine

        self.worker_addr = f"ipc://{address}"
        self.proxy_addr = f"ipc://{proxy_addr}"
        self.ctx = zmq.asyncio.Context()
        self.from_proxy = self.ctx.socket(zmq.constants.PULL)
        self.from_proxy.bind(self.worker_addr)
        self.to_proxy = self.ctx.socket(zmq.constants.PUSH)
        self.to_proxy.connect(self.proxy_addr)

        self.decoder_generate = msgspec.msgpack.Decoder(GenerationRequest)
        self.decoder_heartbeat = msgspec.msgpack.Decoder(HeartbeatRequest)
        self.decoder_abort = msgspec.msgpack.Decoder(GenerationRequest)
        self.decoder_profile = msgspec.msgpack.Decoder(ProfileRequest)
        self.encoder = msgspec.msgpack.Encoder()

        self.running_requests: set[asyncio.Task] = set()

    def shutdown(self):
        self.ctx.destroy()

        for running_request in self.running_requests:
            running_request.cancel()

        socket_path = self.worker_addr.replace("ipc://", "")
        if os.path.exists(socket_path):
            os.remove(socket_path)

    async def run_busy_loop(self):
        logger.info("DisaggWorker is ready To handle requests.")

        poller = zmq.asyncio.Poller()
        poller.register(self.from_proxy, zmq.POLLIN)

        while True:
            req_type, req_data = await self.from_proxy.recv_multipart()
            await self._handle_request(req_type, req_data)

    async def _handle_request(self, req_type: bytes, req_data: bytes):
        if req_type == RequestType.ENCODE:
            gen_req = self.decoder_generate.decode(req_data)
            gen_req.sampling_params.max_tokens = 1
            await self._encode_handler(gen_req)
        elif req_type == RequestType.GENERATION:
            gen_req = self.decoder_generate.decode(req_data)
            await self._generation_handler(gen_req)
        elif req_type == RequestType.ABORT:
            gen_req = self.decoder_abort.decode(req_data)
            await self._abort_handler(gen_req)
        elif req_type == RequestType.HEARTBEAT:
            hb_req = self.decoder_heartbeat.decode(req_data)
            await self._heartbeat_handler(hb_req)
        elif req_type == RequestType.START_PROFILE:
            profile_req = self.decoder_profile.decode(req_data)
            await self._start_profile_handler(profile_req)
        elif req_type == RequestType.STOP_PROFILE:
            profile_req = self.decoder_profile.decode(req_data)
            await self._stop_profile_handler(profile_req)
        else:
            raise Exception(f"Unknown Request Type: {req_type.decode()}.")

    async def _encode_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.ENCODE, b))
        )
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _generation_handler(self, req: GenerationRequest):
        task = asyncio.create_task(
            self._generate(req, lambda b: (ResponseType.GENERATION, b))
        )
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _abort_handler(self, req: GenerationRequest):
        self.engine.abort(request_id=req.request_id)

    async def _heartbeat_handler(self, req: HeartbeatRequest):
        msg = (
            ResponseType.HEARTBEAT,
            self.encoder.encode(
                HeartbeatResponse(request_id=req.request_id, status="OK")
            ),
        )
        await self.to_proxy.send_multipart(msg, copy=False)

    async def _start_profile_handler(self, req: ProfileRequest):
        """Handle start profiling request by delegating to the engine."""
        try:
            await self.engine.start_profile()
            logger.info("Profiling started for request %s", req.request_id)
        except Exception as e:
            logger.exception("Failed to start profiling for request %s", req.request_id)
            # Send failure response back to proxy
            failure_resp = FailureResponse(
                request_id=req.request_id, 
                error_message=f"Failed to start profiling: {str(e)}"
            )
            response_bytes = self.encoder.encode(failure_resp)
            msg = (ResponseType.FAILURE, response_bytes)
            await self.to_proxy.send_multipart(msg, copy=False)

    async def _stop_profile_handler(self, req: ProfileRequest):
        """Handle stop profiling request by delegating to the engine."""
        try:
            await self.engine.stop_profile()
            logger.info("Profiling stopped for request %s", req.request_id)
        except Exception as e:
            logger.exception("Failed to stop profiling for request %s", req.request_id)
            # Send failure response back to proxy
            failure_resp = FailureResponse(
                request_id=req.request_id,
                error_message=f"Failed to stop profiling: {str(e)}"
            )
            response_bytes = self.encoder.encode(failure_resp)
            msg = (ResponseType.FAILURE, response_bytes)
            await self.to_proxy.send_multipart(msg, copy=False)

    async def _generate(
        self,
        req: GenerationRequest,
        make_msg_func,
    ):
        request_id = req.request_id

        try:
            prompt_payload: dict[str, Any] = {"prompt": req.prompt}
            if req.multi_modal_data is not None:
                prompt_payload["multi_modal_data"] = _decode_mm_data(
                    req.multi_modal_data
                )

            generator = self.engine.generate(
                prompt=prompt_payload,
                sampling_params=req.sampling_params,
                request_id=request_id,
            )

            async for request_output in generator:
                response = GenerationResponse.from_request_output(
                    request_output
                )

                response_bytes = self.encoder.encode(response)
                msg = make_msg_func(response_bytes)
                await self.to_proxy.send_multipart(msg, copy=False)
        except Exception as e:
            logger.exception("Generation failed for request %s", request_id)
            failure_resp = FailureResponse(
                request_id=request_id, error_message=str(e) or type(e).__name__
            )
            response_bytes = self.encoder.encode(failure_resp)
            msg = (ResponseType.FAILURE, response_bytes)
            await self.to_proxy.send_multipart(msg, copy=False)


def _decode_mm_data(mm_data: dict[str, Any]) -> dict[str, Any]:
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]
    decoded_list: list[NDArray[Any]] = []
    for img in images:
        if img["type"] == "ndarray":
            decoded_img = np.frombuffer(
                bytes(img["data"]), dtype=img["dtype"]
            ).reshape(img["shape"])
            decoded_list.append(decoded_img)
    result_images: list[NDArray[Any]] | NDArray[Any]
    if len(decoded_list) == 1:
        result_images = decoded_list[0]
    else:
        result_images = decoded_list
    return {"image": result_images}
