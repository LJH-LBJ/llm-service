# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import asyncio
import os
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional, Union

import msgspec
import numpy as np
import zmq
import zmq.asyncio

from vllm.config import ModelConfig, VllmConfig
from llm_service.protocol.protocol import (
    FailureResponse,
    GenerationRequest,
    GenerationResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    MetricsRequest,
    MetricsResponse,
    RequestType,
    ResponseType,
    ServerType,
)
from llm_service.request_stats import RequestStatsMonitor
from llm_service.routing_logic import RandomRouter, RoutingInterface
from llm_service.service_discovery import HealthCheckServiceDiscovery
from llm_service.metrics_reporter import MetricsReporter

from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Device

TIMECOUNT_ENABLED = os.getenv("TIMECOUNT_ENABLED", "0") in ("1", "true", "True")

logger = init_logger(__name__)


class Proxy(EngineClient):
    """
    Proxy
    """

    def __init__(
        self,
        proxy_addr: str,
        encode_addr_list: list[str],
        pd_addr_list: list[str],
        model_name: str,
        router: type[RoutingInterface] = RandomRouter,
        enable_health_monitor=True,
        health_check_interval=10,
        health_threshold=3,
    ):
        self.queues: dict[str, asyncio.Queue] = {}

        self.encoder = msgspec.msgpack.Encoder()

        self.ctx = zmq.asyncio.Context()
        self.proxy_addr = f"ipc://{proxy_addr}"
        self.encode_addr_list = [f"ipc://{addr}" for addr in encode_addr_list]
        self.pd_addr_list = [f"ipc://{addr}" for addr in pd_addr_list]
        self.to_encode_sockets = []
        for addr in self.encode_addr_list:
            socket = self.ctx.socket(zmq.constants.PUSH)
            socket.connect(addr)
            self.to_encode_sockets.append(socket)
        self.to_pd_sockets = []
        for addr in self.pd_addr_list:
            socket = self.ctx.socket(zmq.constants.PUSH)
            socket.connect(addr)
            self.to_pd_sockets.append(socket)

        self.enable_health_monitor = enable_health_monitor
        self.health_check_interval = health_check_interval
        self.health_threshold = health_threshold
        self.encode_service_discovery = HealthCheckServiceDiscovery(
            server_type=ServerType.E_INSTANCE,
            instances=list(range(len(self.encode_addr_list))),
            enable_health_monitor=self.enable_health_monitor,
            health_check_interval=self.health_check_interval,
            health_threshold=self.health_threshold,
            health_check_func=self.check_health,
        )

        self.pd_service_discovery = HealthCheckServiceDiscovery(
            server_type=ServerType.PD_INSTANCE,
            instances=list(range(len(self.pd_addr_list))),
            enable_health_monitor=self.enable_health_monitor,
            health_check_interval=self.health_check_interval,
            health_threshold=self.health_threshold,
            health_check_func=self.check_health,
        )
        self.pd_metrics_logger = MetricsReporter(
            server_type=ServerType.PD_INSTANCE,
            instances=list(range(len(self.pd_addr_list))),
            get_metrics_func=self.get_metrics,
        )
        self.encoder_metrics_logger = MetricsReporter(
            server_type=ServerType.E_INSTANCE,
            instances=list(range(len(self.encode_addr_list))),
            get_metrics_func=self.get_metrics,
        )
        self.encode_request_stats_monitor = RequestStatsMonitor(
            list(range(len(self.encode_addr_list)))
        )
        self.pd_request_stats_monitor = RequestStatsMonitor(
            list(range(len(self.pd_addr_list)))
        )
        self.encode_router = router()
        self.pd_router = router()

        self.output_handler: Optional[asyncio.Task] = None

        # Dummy: needed for EngineClient Protocol.
        self.model_config = ModelConfig(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="auto",
            task="generate",
            seed=42,
        )

        self.proxy_to_pd_time: list[float] = [0.0, 0.0]  # count, total time
        self.proxy_to_encode_time: list[float] = [0.0, 0.0]  # count, total time

    def shutdown(self):
        self.ctx.destroy()
        if (task := self.output_handler) is not None:
            task.cancel()

        socket_path = self.proxy_addr.replace("ipc://", "")
        if os.path.exists(socket_path):
            os.remove(socket_path)

    async def _run_encode(
        self,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ) -> None:
        """
        Send the encode request to one encoder worker.
        The encoder worker is selected based on hashing the request ID.
        """
        if not self.to_encode_sockets:
            raise RuntimeError(
                "No encode workers configured: encode_addr_list is empty."
            )

        try:
            payload = self.encoder.encode(request)
        except Exception as e:
            raise RuntimeError("Failed to serialize GenerationRequest") from e

        msg = (RequestType.ENCODE, payload)
        health_endpoints = self.encode_service_discovery.get_health_endpoints()
        request_stats = self.encode_request_stats_monitor.get_request_stats()
        idx = self.encode_router.route_request(health_endpoints, request_stats)
        self.encode_request_stats_monitor.on_new_request(
            idx, request_id=request.request_id
        )
        try:
            socket = self.to_encode_sockets[idx]
            if TIMECOUNT_ENABLED:
                proxy_to_encode_time_start = time.perf_counter()
            await socket.send_multipart(msg, copy=False)
            response = await q.get()
            if (
                TIMECOUNT_ENABLED
                and isinstance(response, GenerationResponse)
                and response.proxy_to_worker_time_end
            ):
                self.proxy_to_encode_time[0] += 1
                self.proxy_to_encode_time[1] += (
                    response.proxy_to_worker_time_end
                    - proxy_to_encode_time_start  # type: ignore
                )

            if isinstance(response, Exception):
                raise response
        finally:
            self.encode_request_stats_monitor.on_request_completed(
                idx, request_id=request.request_id
            )

    async def _run_pd(
        self,
        request: GenerationRequest,
        q: asyncio.Queue[Union[Exception, GenerationResponse]],
    ):
        """
        Send the generation request to a PD worker and yield its response.
        The PD worker is selected based on hashing the request ID.
        """
        if not self.to_pd_sockets:
            raise RuntimeError(
                "No PD workers configured: pd_addr_list is empty."
            )

        try:
            payload = self.encoder.encode(request)
        except Exception as e:
            raise RuntimeError("Failed to serialize GenerationRequest") from e

        msg = (RequestType.GENERATION, payload)
        health_endpoints = self.pd_service_discovery.get_health_endpoints()
        request_stats = self.pd_request_stats_monitor.get_request_stats()
        idx = self.pd_router.route_request(health_endpoints, request_stats)
        self.pd_request_stats_monitor.on_new_request(
            idx, request_id=request.request_id
        )

        try:
            socket = self.to_pd_sockets[idx]
            if TIMECOUNT_ENABLED:
                proxy_to_pd_time_start = time.perf_counter()
            await socket.send_multipart(msg, copy=False)
            finished = False
            while not finished:
                response = await q.get()
                if isinstance(response, Exception):
                    raise response
                if (
                    TIMECOUNT_ENABLED
                    and isinstance(response, GenerationResponse)
                    and response.proxy_to_worker_time_end
                ):
                    self.proxy_to_pd_time[0] += 1
                    self.proxy_to_pd_time[1] += (
                        response.proxy_to_worker_time_end
                        - proxy_to_pd_time_start  # type: ignore
                    )
                finished = response.finish_reason is not None
                yield response
        finally:
            self.pd_request_stats_monitor.on_request_completed(
                idx, request_id=request.request_id
            )

    def _to_request_output(self, resp: GenerationResponse) -> RequestOutput:
        """Convert a PD/Generate response to vLLM RequestOutput.

        This creates a single CompletionOutput. If the response includes
        text/token_ids attributes, they are used; otherwise defaults are used.
        """
        text = getattr(resp, "text", "")
        token_ids = getattr(resp, "token_ids", [])

        completion = CompletionOutput(
            index=0,
            text=text,
            token_ids=token_ids,
            cumulative_logprob=None,
            logprobs=None,
            finish_reason=resp.finish_reason,
            stop_reason=resp.stop_reason,
        )

        return RequestOutput(
            request_id=resp.request_id,
            prompt=None,
            prompt_token_ids=resp.prompt_token_ids,
            prompt_logprobs=None,
            outputs=[completion],
            finished=resp.finish_reason is not None,
        )

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ):
        # lazy initialization
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler()
            )
        if self.encode_service_discovery.should_launch_health_monitor():
            self.encode_service_discovery.launch_health_monitor()
        if self.pd_service_discovery.should_launch_health_monitor():
            self.pd_service_discovery.launch_health_monitor()

        if not request_id:
            request_id = uuid.uuid4().hex

        q: asyncio.Queue = asyncio.Queue()
        if request_id in self.queues:
            raise ValueError(f"Request id {request_id} already running.")
        else:
            self.queues[request_id] = q

        # Support both raw string prompts and dict prompts with multimodal data
        prompt_text = prompt["prompt"] if isinstance(prompt, dict) else prompt

        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt_text,
            sampling_params=sampling_params,
        )

        try:
            # need to validate to avoid decode failed later
            req_dict = msgspec.to_builtins(request)
            request = msgspec.convert(req_dict, GenerationRequest, strict=True)

            if _has_mm_data(prompt):
                request.multi_modal_data = _encode_mm_data(
                    prompt["multi_modal_data"]
                )
                await self._run_encode(request, q)

            # TODO: support pd separation
            async for pd_response in self._run_pd(request, q):
                yield self._to_request_output(pd_response)
        except msgspec.ValidationError as e:
            raise RuntimeError(f"Invalid Parameters: {e}.") from e
        finally:
            self.queues.pop(request_id, None)

    async def abort_requests_from_unhealth_endpoints(
        self, server_type, unhealth_endpoints, request_stats_monitor
    ) -> None:
        request_stats = request_stats_monitor.get_request_stats()

        async def fail_request(req_id, iid):
            if req_id in self.queues:
                await self.queues[req_id].put(
                    RuntimeError(
                        f"{server_type} instance {iid} is unhealthy, "
                        f"so abort its request {req_id}."
                    )
                )

        tasks = [
            asyncio.create_task(fail_request(req_id, iid))
            for iid in unhealth_endpoints
            for req_id in request_stats.get(iid).in_flight_requests
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_output_handler(self) -> None:
        """Background task to pull responses and dispatch to request queues.

        Binds a PULL socket on proxy_addr and receives multipart messages of
        the form (response_type, payload). Decodes payload into a
        GenerationResponse and enqueues it into the corresponding request queue
        keyed by request_id.
        """
        socket: Optional[zmq.asyncio.Socket] = None
        decoder = msgspec.msgpack.Decoder(GenerationResponse)
        failure_decoder = msgspec.msgpack.Decoder(FailureResponse)
        heartbeat_decoder = msgspec.msgpack.Decoder(HeartbeatResponse)
        metrics_decoder = msgspec.msgpack.Decoder(MetricsResponse)
        try:
            socket = self.ctx.socket(zmq.constants.PULL)
            socket.bind(self.proxy_addr)
            timeout = self.health_check_interval * self.health_threshold / 2

            while True:
                encode_unhealths = (
                    self.encode_service_discovery.get_unhealth_endpoints()
                )
                pd_unhealths = (
                    self.pd_service_discovery.get_unhealth_endpoints()
                )
                tasks = []
                if encode_unhealths:
                    tasks.append(
                        self.abort_requests_from_unhealth_endpoints(
                            server_type=ServerType.E_INSTANCE,
                            unhealth_endpoints=encode_unhealths,
                            request_stats_monitor=self.encode_request_stats_monitor,
                        )
                    )
                if pd_unhealths:
                    tasks.append(
                        self.abort_requests_from_unhealth_endpoints(
                            server_type=ServerType.PD_INSTANCE,
                            unhealth_endpoints=pd_unhealths,
                            request_stats_monitor=self.pd_request_stats_monitor,
                        )
                    )
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Check if the engine is alive:
                if not await socket.poll(timeout=timeout):
                    continue
                resp_type, payload = await socket.recv_multipart()

                # Decode response according to its type.
                resp: Union[
                    GenerationResponse,
                    HeartbeatResponse,
                    FailureResponse,
                    MetricsResponse,
                ]
                if resp_type in (ResponseType.GENERATION, ResponseType.ENCODE):
                    resp = decoder.decode(payload)
                elif resp_type == ResponseType.HEARTBEAT:
                    resp = heartbeat_decoder.decode(payload)
                elif resp_type == ResponseType.FAILURE:
                    resp = failure_decoder.decode(payload)
                elif resp_type == ResponseType.METRICS:
                    resp = metrics_decoder.decode(payload)
                else:
                    raise RuntimeError(
                        f"Unknown response type from worker: {resp_type.decode()}"
                    )

                if resp.request_id not in self.queues:
                    logger.warning(
                        "Request %s may have been aborted, ignore response.",
                        resp.request_id,
                    )
                elif isinstance(resp, FailureResponse):
                    self.queues[resp.request_id].put_nowait(
                        RuntimeError(f"Request error: {resp.error_message}")
                    )
                else:
                    self.queues[resp.request_id].put_nowait(resp)

        except Exception as e:
            # TODO: maybe there is a more fine-grained way to handle errors.
            # For now, if there is any error, we terminate all requests.
            for q in self.queues.values():
                q.put_nowait(e)
        finally:
            if socket is not None:
                socket.close(linger=0)

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError

    async def abort(self, request_id: str) -> None:
        raise NotImplementedError

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        raise NotImplementedError

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_input_preprocessor(self) -> InputPreprocessor:
        raise NotImplementedError

    async def get_tokenizer(self) -> AnyTokenizer:
        raise NotImplementedError

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        pass

    async def check_health(self, server_type: ServerType, id: int):
        request_id = str(uuid.uuid4())
        request = HeartbeatRequest(request_id=request_id)
        q: asyncio.Queue = asyncio.Queue()
        self.queues[request_id] = q
        try:
            payload = self.encoder.encode(request)
            msg = (RequestType.HEARTBEAT, payload)
            if server_type == ServerType.PD_INSTANCE:
                socket = self.to_pd_sockets[id]
            else:
                socket = self.to_encode_sockets[id]

            await socket.send_multipart(msg, copy=False)
            response = await q.get()
            if (
                isinstance(response, HeartbeatResponse)
                and response.status == "OK"
            ):
                return True
            elif isinstance(response, Exception):
                raise response
            else:
                return False

        except Exception as e:
            raise RuntimeError(
                f"Health check failed for {server_type} {id}, exception: {e}"
            ) from e
        finally:
            self.queues.pop(request_id, None)

    async def get_metrics(self, server_type: ServerType, id: int):
        request_id = str(uuid.uuid4())
        request = MetricsRequest(request_id=request_id)
        q: asyncio.Queue = asyncio.Queue()
        self.queues[request_id] = q
        try:
            payload = self.encoder.encode(request)
            msg = (RequestType.METRICS, payload)
            if server_type == ServerType.PD_INSTANCE:
                socket = self.to_pd_sockets[id]
            else:
                socket = self.to_encode_sockets[id]

            await socket.send_multipart(msg, copy=False)
            response = await q.get()
            # calculate proxy to pd/encode time
            if (
                isinstance(response, MetricsResponse)
                and response.metrics is not None
            ):
                # calculate proxy to pd/encode time average
                # add to metrics
                proxy2pd_avg = (
                    self.proxy_to_pd_time[1] / self.proxy_to_pd_time[0]
                    if self.proxy_to_pd_time[0] > 0
                    else 0.0
                )
                proxy2encode_avg = (
                    self.proxy_to_encode_time[1] / self.proxy_to_encode_time[0]
                    if self.proxy_to_encode_time[0] > 0
                    else 0.0
                )
                response.metrics.update(
                    {
                        "proxy_to_pd_time_avg": proxy2pd_avg,
                        "proxy_to_encode_time_avg": proxy2encode_avg,
                    }
                )
                return response.metrics
            elif isinstance(response, Exception):
                raise response
            else:
                return None

        except Exception as e:
            raise RuntimeError(
                "Get metrics failed for %s %s, exception: %s",
                server_type,
                id,
                e,
            ) from e
        finally:
            self.queues.pop(request_id, None)

    async def start_profile(self) -> None:
        raise NotImplementedError

    async def stop_profile(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache(self, device: Optional[Device] = None) -> None:
        raise NotImplementedError

    async def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    @property
    def errored(self) -> bool:
        return False

    def dead_error(self) -> Exception:
        return Exception("PDController has failed.")

    def is_running(self) -> bool:
        return True

    def is_stopped(self) -> bool:
        return False

    async def reset_mm_cache(self) -> None:
        raise NotImplementedError


def _has_mm_data(prompt: PromptType) -> bool:
    if isinstance(prompt, dict):
        return "multi_modal_data" in prompt
    return False


def _encode_mm_data(mm_data: dict[str, Any]) -> dict[str, Any]:
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]
    encoded_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            encoded_img = {
                "type": "ndarray",
                "data": img.tobytes(),
                "shape": img.shape,
                "dtype": str(img.dtype),
            }
            encoded_images.append(encoded_img)
    return {"image": encoded_images}
