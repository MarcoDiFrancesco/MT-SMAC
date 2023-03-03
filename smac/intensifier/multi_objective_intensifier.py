# TODO does this work for multi-fidelity?
# Yes, then pass a pareto front calculation function to the abstract intensifier instead of subclassing it

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration

import smac
from smac.callback import Callback
from smac.constants import MAXINT
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo
from smac.runhistory.dataclasses import (
    InstanceSeedBudgetKey,
    InstanceSeedKey,
    TrajectoryItem,
    TrialValue,
)
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.configspace import get_config_hash, print_config_changes
from smac.utils.logging import get_logger
from smac.utils.pareto_front import calculate_pareto_front, sort_by_crowding_distance
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.intensifier.intensifier import Intensifier


__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

# TODO add minimum population size?

class MOIntensifierMixin(object):
    def _calculate_pareto_front(
        self,
        runhistory: RunHistory,
        configs: list[Configuration],
        config_instance_seed_budget_keys: list[list[InstanceSeedBudgetKey]],
    ) -> list[Configuration]:
    # TODO use fast non dominance sorting
        return calculate_pareto_front(
            runhistory=runhistory,
            configs=configs,
            config_instance_seed_budget_keys=config_instance_seed_budget_keys,
        )

    def _remove_incumbent(self, config: Configuration, previous_incumbent_ids: list[int], new_incumbent_ids: list[int]) -> None:
        # TODO adjust
        raise NotImplementedError

    def _cut_incumbents(self, incumbent_ids: list[int], all_incumbent_isb_keys: list[list[InstanceSeedBudgetKey]]) -> list[int]:
        #TODO JG sort by hypervolume
        new_incumbents = sort_by_crowding_distance(self.runhistory, incumbent_ids, all_incumbent_isb_keys)
        new_incumbents = new_incumbents[: self._max_incumbents]

        logger.info(
            f"Removed one incumbent using their reduction in hypervolume because more than {self._max_incumbents} are "
            "available."
        )

        return new_incumbents

    def get_instance_seed_budget_keys(
        self, config: Configuration, compare: bool = False
    ) -> list[InstanceSeedBudgetKey]:
        """Returns the instance-seed-budget keys for a given configuration. This method is *used for
        updating the incumbents* and might differ for different intensifiers. For example, if incumbents should only
        be compared on the highest observed budgets.
        """
        return self.runhistory.get_instance_seed_budget_keys(config, highest_observed_budget_only=True)

class MOIntensifier(Intensifier, MOIntensifierMixin):
    pass

class MOSuccessiveHalving(SuccessiveHalving, MOIntensifierMixin):
    pass

class MOHyperband(Hyperband, MOIntensifierMixin):
    pass