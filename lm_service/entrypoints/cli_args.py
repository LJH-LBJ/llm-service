# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl
from collections.abc import Sequence
from dataclasses import field
from typing import Literal

from pydantic.dataclasses import dataclass

from lm_service.logger_utils import init_logger

from vllm.config import config
from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.utils.argparse_utils import FlexibleArgumentParser


logger = init_logger(__name__)

def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    """Create the CLI argument parser used by the OpenAI API server.
    """
    parser.add_argument(
        "--proxy-addr",
        required=False,
        help="The address of the proxy server.",
    )
    parser.add_argument(
        "--encoder-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the encoder.",
    )
    parser.add_argument(
        "--pd-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the pd.",
    )
    parser.add_argument(
        "--p-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the prefill.",
    )
    parser.add_argument(
        "--d-addr-list",
        required=False,
        nargs="+",
        help="List of addresses for the decode.",
    )
    parser.add_argument(
        "--transfer-protocol",
        type=str,
        default="ipc",
        choices=["ipc", "tcp"],
        help="ZMQ transfer protocol, whether ZMQ uses IPC or TCP connection",
    )
    parser.add_argument(
        "--metastore-client-config",
        type=json.loads,
        default=None,
        help="Enable metastore client config.",
    )
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--image-path", required=True, help="Path to the image")

    return parser