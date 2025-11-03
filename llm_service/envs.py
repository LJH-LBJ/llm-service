# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Callable

_TRUE_VALUES = {"1", "true", "t", "y", "yes", "on"}
_FALSE_VALUES = {"0", "false", "f", "n", "no", "off"}

# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    "TIMECOUNT_ENABLED": lambda: os.getenv("TIMECOUNT_ENABLED", "0").lower()
    in _TRUE_VALUES,
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
