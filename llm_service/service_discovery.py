# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import asyncio
import time
from abc import ABC, abstractmethod

from llm_service.protocol.protocol import ServerType
from vllm.logger import init_logger

logger = init_logger(__name__)


class ServiceDiscovery(ABC):
    @abstractmethod
    def get_health_endpoints(self) -> list[int]:
        """
        Retrieve a list of available instances for the given service name.

        Args:
            service_name (str): The name of the service to discover.
        Returns:
            list[int]: A list of available instance IDs.
        """
        pass

    @abstractmethod
    def get_unhealth_endpoints(self) -> list[int]:
        """
        Retrieve a list of available instances for the given service name.

        Args:
            service_name (str): The name of the service to discover.

        Returns:
            list[int]: A list of available instance IDs.
        """
        pass


class HealthCheckServiceDiscovery(ServiceDiscovery):
    def __init__(
        self,
        server_type: ServerType,
        instances: list[int],
        enable_health_monitor: bool,
        health_check_interval: float,
        health_threshold: int,
        health_check_func,
    ):
        self.server_type = server_type
        self._instances = {iid: True for iid in instances}
        self._cached_health_instances = [iid for iid in instances]
        self._cached_unhealth_instances: list[int] = []
        self.enable_health_monitor = enable_health_monitor
        self._health_check_interval = health_check_interval
        self._health_threshold = health_threshold
        self._succ_count = {iid: 0 for iid in instances}
        self._fail_count = {iid: 0 for iid in instances}
        self._health_check_func = health_check_func
        self._health_monitor_handler = None

    def should_launch_health_monitor(self) -> bool:
        return (
            self.enable_health_monitor and self._health_monitor_handler is None
        )

    def launch_health_monitor(self):
        self._health_monitor_handler = asyncio.create_task(
            self.run_health_check_loop()
        )
        logger.info("Health monitor for %s launched.", self.server_type)

    def get_health_endpoints(self) -> list[int]:
        return self._cached_health_instances

    def get_unhealth_endpoints(self) -> list[int]:
        return self._cached_unhealth_instances

    async def run_health_check_loop(self):
        while True:
            start_time = time.monotonic()
            tasks = [
                asyncio.create_task(
                    asyncio.wait_for(
                        self._health_check_func(self.server_type, iid),
                        timeout=self._health_check_interval,
                    )
                )
                for iid in self._instances
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for iid, result in zip(self._instances.keys(), results):
                if isinstance(result, bool) and result:
                    self._update_health_counts(iid, True)
                else:
                    self._update_health_counts(iid, False)
                    logger.warning(
                        "Health check for %s %s failed, reason is (%s).",
                        self.server_type,
                        iid,
                        "timeout"
                        if isinstance(result, asyncio.TimeoutError)
                        else result,
                    )

            self._update_health_status()

            elapsed = time.monotonic() - start_time
            sleep_time = max(0, self._health_check_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def _update_health_counts(self, iid: int, is_succ: bool):
        if is_succ:
            self._succ_count[iid] = min(
                self._health_threshold, self._succ_count.get(iid, 0) + 1
            )
            self._fail_count[iid] = 0
        else:
            self._fail_count[iid] = min(
                self._health_threshold, self._fail_count.get(iid, 0) + 1
            )
            self._succ_count[iid] = 0

    def _update_health_status(self):
        for iid in self._instances:
            if (
                self._instances[iid]
                and self._fail_count.get(iid, 0) >= self._health_threshold
            ):
                self._instances[iid] = False
                logger.info(
                    "Instance %s %s marked as unhealthy.", self.server_type, iid
                )
            elif (
                not self._instances[iid]
                and self._succ_count.get(iid, 0) >= self._health_threshold
            ):
                self._instances[iid] = True
                logger.info(
                    "Instance %s %s marked as healthy.", self.server_type, iid
                )

        self._cached_health_instances = [
            iid for iid, healthy in self._instances.items() if healthy
        ]
        self._cached_unhealth_instances = [
            iid for iid, healthy in self._instances.items() if not healthy
        ]

class MetricsServiceDiscovery():
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
            "Engine %03d: "
            "Avg encoder consume requests: %.3f ms, "
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
        for iid, result in zip(self._instances.keys(), results):
            if isinstance(result, dict):

                msg = log_msg % (
                    result.get("engine_index", 0),
                    result.get("encoder_consume_time", 0.0),
                    result.get("e2e_time_requests", 0.0),
                    result.get("queue_time_requests", 0.0),
                    result.get("prefill_time_requests", 0.0),
                    result.get("mean_time_per_output_token_requests", 0.0),
                    result.get("time_to_first_token", 0.0),
                    result.get("proxy_to_encode_time_avg", 0.0) \
                        if self.server_type== ServerType.E_INSTANCE \
                            else result.get("proxy_to_pd_time_avg", 0.0)
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
        for iid, metric in metrics.items():
            logger.info("Metrics for %s %d: %s", self.server_type, iid, metric)