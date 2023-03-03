from __future__ import annotations

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


from abc import ABC, abstractmethod
from typing import Any, Iterator

import time
import traceback

import numpy as np
from ConfigSpace import Configuration

from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.scenario import Scenario
from smac.utils.logging import get_logger
from smac.runner.target_function_script_runner import TargetFunctionScriptRunner

logger = get_logger(__name__)

class ACLibRunner(TargetFunctionScriptRunner):
    def __init__(self,
                 target_function: str,
                 scenario: Scenario,
                 required_arguments: list[str] = [],
                 target_function_arguments: dict[str, str] | None = None,
                 ):

        self._target_function_arguments = target_function_arguments

        super().__init__(target_function, scenario, required_arguments)
    def __call__(self, algorithm_kwargs: dict[str, Any]) -> tuple[str, str]:
        # kwargs has "instance", "seed" and "budget" --> translate those

        cmd = self._target_function.split(" ")
        if self._target_function_arguments is not None:
            for k, v in self._target_function_arguments.items():
                cmd += [f"--{k}={v}"]

        if self._scenario.trial_walltime_limit is not None:
            cmd += [f"--cutoff={self._scenario.trial_walltime_limit}"]

        config = ["--config"]

        for k, v in algorithm_kwargs.items():
            v = str(v)
            k = str(k)

            # Let's remove some spaces
            v = v.replace(" ", "")

            if k in ["instance", "seed"]:
                cmd += [f"--{k}={v}"]
            elif k == "instance_features":
                continue
            else:
                config += [k, v]

        cmd += config

        logger.debug(f"Calling: {' '.join(cmd)}")
        print(f"Calling: {' '.join(cmd)}")
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        output, error = p.communicate()

        logger.debug("Stdout: %s" % output)
        logger.debug("Stderr: %s" % error)

        return output, error

