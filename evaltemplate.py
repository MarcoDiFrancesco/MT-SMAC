# %%
import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from smac import Scenario

# %% [markdown]
# ## Creating some mock problems
# 

# %%
def DLTZ1(cfg, seed=0):
    """Classic MO continuous optimization problem
    From: A New Distributed Evolutionary Computation Technique for Multi-Objective Optimization
    By: Islam et al., 2010
    https://arxiv.org/pdf/1304.2543.pdf

    DLTZ1 about:
    - Known Pareto front: It is designed to have a known Pareto front (the set of all non-dominated solutions)
    and a known set of trade-offs between conflicting objectives, making it easier to
    evaluate how well an optimization algorithm is performing.
    - Multi-Modality: The problem is multi-modal, meaning it has multiple local optima.
    - Non-Convex Pareto Front: The Pareto front is non-convex, meaning that there are regions
    of the objective space where improving one objective leads to a deterioration in another,
    and these trade-offs are not linear.

    Parameters

    ----------
    cfg: Configuration

    Returns
    -------

    """
    x = cfg.get_array() if isinstance(cfg, Configuration) else cfg
    objectives = 2
    n = len(x)  # Decision space
    k = n - objectives + 1

    xm = x[objectives - 1 :]
    g = 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))

    y = np.ones(objectives) * (0.5 * (1 + g))

    prod_xi = 1

    for i in range(objectives - 1, 0, -1):
        y[i] *= prod_xi * (1 - x[objectives - (i + 1)])
        prod_xi *= x[objectives - (i + 1)]

    y[0] *= prod_xi

    return {"y{}".format(i + 1): obj for i, obj in enumerate(y)}


def MMF4(cfg, seed=0):
    """Classic MO continuous optimization problem

    Parameters
    ----------
    cfg: Configuration

    Returns
    -------

    """
    x = cfg.get_array() if isinstance(cfg, Configuration) else cfg
    objectives = 2
    y = [0, 0]
    y[0] = x[0]
    if x[1] >= 1:
        x[1] -= 1
    y[0] = abs(x[0])
    y[1] = 1.0 - x[0] ** 2 + 2 * (x[1] - np.sin(np.pi * np.abs(x[0]))) ** 2

    return {"y{}".format(i + 1): -obj for i, obj in enumerate(y)}

# %%
MMF4([0.5, 9, 1, 0, 0])
DLTZ1(np.array([0, 0.6, 0, 0.9, 0]))

# %% [markdown]
# ## Defining the scenarios
# 

# %%
trials = 100

# %%
def DLTZ1_scenario():
    cs = ConfigurationSpace(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "x3": (0.0, 1.0),
            "x4": (0.0, 1.0),
            "x5": (0.0, 1.0),
        }
    )

    # Scenario object
    scenario = Scenario(
        configspace=cs,
        deterministic=True,
        objectives=["y1", "y2"],
        crash_cost=[0, 0],
        n_trials=trials,
        seed=1,
        n_workers=1,
    )

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    config = cs.sample_configuration()
    print(f"{DLTZ1(config)=}")

    return scenario, DLTZ1, ["y1", "y2"]


def MMF4_scenario():
    cs = ConfigurationSpace(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "x3": (0.0, 1.0),
            "x4": (0.0, 1.0),
            "x5": (0.0, 1.0),
        }
    )

    # Scenario object
    scenario = Scenario(
        configspace=cs,
        deterministic=True,
        objectives=["y1", "y2"],
        crash_cost=[0, 0],
        n_trials=trials,
        seed=1,
        n_workers=1,
    )

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    config = cs.sample_configuration()
    print(f"{MMF4(config)=}")

    return scenario, MMF4, ["y1", "y2"]


DLTZ1_scenario()

# %% [markdown]
# ## Set up the configurators
# 

# %%
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import AlgorithmConfigurationFacade as ACFacade
from smac.initial_design import RandomInitialDesign
from smac.model.random_forest import RandomForest

# %%
scenario, target_function, objectives = DLTZ1_scenario()

# %%
# ParEGO
smac_pe = ACFacade(
    scenario=scenario,
    target_function=target_function,
    overwrite=True,
    initial_design=RandomInitialDesign(scenario, n_configs=2),
    model=RandomForest(scenario.configspace, log_y=False),
)

# smac_pe.optimize()

# %%
from smac.facade.multi_objective_facade import MultiObjectiveFacade as MOFacade

# MO-SMAC
smac_phvi = MOFacade(
    scenario=scenario,
    target_function=target_function,
    overwrite=True,
    initial_design=RandomInitialDesign(scenario, n_configs=2),
    model=MOFacade.get_model(scenario),
)

# smac_phvi.optimize()

# %%
from smac.model.random_forest.random_forest_mo import RandomForestMO
from smac.facade.algorithm_configuration_facade_mo import AlgorithmConfigurationFacadeMO as ACFacadeMO

smac_xgb = ACFacadeMO(
    scenario=scenario,
    target_function=target_function,
    overwrite=True,
    initial_design=RandomInitialDesign(scenario, n_configs=2),
    # model=RandomForestMO(scenario.configspace),
    model=ACFacadeMO.get_model(scenario),
)

smac_xgb.optimize()

# %% [markdown]
# ## Plotting
# 

# %%
configurators = {
    "ParEGO": smac_pe,
    "MO-SMAC": smac_phvi,
    "XGB": smac_xgb,
}

for configname, configurator in configurators.items():
    # Get all configs
    configs = configurator.runhistory.get_configs_per_budget()
    configurator_costs = [configurator.runhistory.average_cost(i, normalize=False) for i in configs]
    plt.scatter(*list(zip(*configurator_costs)), label=configname, marker="x")
plt.legend()
plt.show()

for configname, configurator in configurators.items():
    # Get incumbent configs
    configs = configurator.runhistory.incumbents
    configurator_costs = [configurator.runhistory.average_cost(i, normalize=False) for i in configs]
    plt.scatter(*list(zip(*configurator_costs)), label=configname, marker="x")
plt.legend()
plt.show()


