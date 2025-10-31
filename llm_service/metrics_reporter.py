# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import asyncio

from llm_service.protocol.protocol import ServerType
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(handler)


class MetricsReporter:
    def __init__(
        self,
        server_type: ServerType,
        instances: list[int],
        get_metrics_func,
    ):
        self.server_type = server_type
        self._instances = {iid: True for iid in instances}
        self._get_metrics_func = get_metrics_func

    async def get_metrics(self) -> None:
        metrics = {}
        tasks = [
            asyncio.create_task(
                asyncio.wait_for(
                    self._get_metrics_func(self.server_type, iid),
                    timeout=1.0,
                )
            )
            for iid in self._instances
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        log_msg = (
            "instances: %03d, "
            "Engine %03d: "
            "Avg e2e time requests: %.3f ms, "
            "Avg queue time requests: %.3f ms, "
            "Avg prefill time requests: %.3f ms, "
            "Avg mean time per output token requests: %.3f ms, "
            "Avg time to first token: %.3f ms, "
        )
        if self.server_type == ServerType.E_INSTANCE:
            log_msg += "Avg proxy to encoder requests: %.3f ms, "
        else:
            log_msg += "Avg proxy to pd requests: %.3f ms, "
        msg = ''
        for iid, result in zip(self._instances.keys(), results):
            if isinstance(result, dict):
                for _, value in result.items():
                    msg = log_msg % (
                        iid,
                        value.get("instance_id", 0),
                        value.get("engine_index", 0),
                        value.get("e2e_time_requests", 0.0),
                        value.get("queue_time_requests", 0.0),
                        value.get("prefill_time_requests", 0.0),
                        value.get("mean_time_per_output_token_requests", 0.0),
                        value.get("time_to_first_token", 0.0),
                        value.get("proxy_to_encode_time_avg", 0.0)
                        if self.server_type == ServerType.E_INSTANCE
                        else value.get("proxy_to_pd_time_avg", 0.0),
                    )

                metrics[iid] = msg
            else:
                logger.warning(
                    "Get metrics for %s %s failed, reason is (%s).",
                    self.server_type,
                    iid,
                    "timeout"
                    if isinstance(result, asyncio.TimeoutError)
                    else result,
                )
        logger.info("Metrics for %s instances:" % self.server_type)
        for iid, metric in metrics.items():
            logger.info(metric)
