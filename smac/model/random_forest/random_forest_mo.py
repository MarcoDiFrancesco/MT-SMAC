from __future__ import annotations
from typing import Any
import xgboost

import numpy as np
from ConfigSpace import ConfigurationSpace
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest
from pyrfr.regression import default_data_container as DataContainer
from xgboost import XGBRegressor

from smac.constants import N_TREES, VERY_SMALL_NUMBER
from smac.model.random_forest import AbstractRandomForest, RandomForest
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RandomForestMO(AbstractRandomForest):
    """Random forest that takes instance features into account.

    Parameters
    ----------
    n_trees : int, defaults to `N_TREES`
        The number of trees in the random forest.
    n_points_per_tree : int, defaults to -1
        Number of points per tree. If the value is smaller than 0, the number of samples will be used.
    ratio_features : float, defaults to 5.0 / 6.0
        The ratio of features that are considered for splitting.
    min_samples_split : int, defaults to 3
        The minimum number of data points to perform a split.
    min_samples_leaf : int, defaults to 3
        The minimum number of data points in a leaf.
    max_depth : int, defaults to 2**20
        The maximum depth of a single tree.
    eps_purity : float, defaults to 1e-8
        The minimum difference between two target values to be considered.
    max_nodes : int, defaults to 2**20
        The maximum total number of nodes in a tree.
    bootstrapping : bool, defaults to True
        Enables bootstrapping.
    log_y: bool, defaults to False
        The y values (passed to this random forest) are expected to be log(y) transformed.
        This will be considered during predicting.
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        objectives,
        n_trees: int = N_TREES,
        n_points_per_tree: int = -1,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        eps_purity: float = 1e-8,
        max_nodes: int = 2**20,
        bootstrapping: bool = True,
        log_y: bool = False,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ) -> None:
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )
        
        self._n_objectives = len(objectives)

        max_features = 0 if ratio_features > 1.0 else max(1, int(len(self._types) * ratio_features))

        self._log_y = log_y
        self._rng = regression.default_random_engine(seed)

        self._n_trees = n_trees
        self._n_points_per_tree = n_points_per_tree
        self._ratio_features = ratio_features
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_depth = max_depth
        self._eps_purity = eps_purity
        self._max_nodes = max_nodes
        self._bootstrapping = bootstrapping

        self._rf = XGBRegressor(
            n_estimators=n_trees,
            # n_points_per_tree
            # ratio_features
            # min_samples_split
            # min_samples_leaf
            max_depth=max_depth,
            # eps_purity
            # max_nodes
            # bootstrapping
            multi_strategy="multi_output_tree",
            tree_method="hist",
        )

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        logger.warning("Code below not checked")

        meta = super().meta
        meta.update(
            {
                "n_trees": self._n_trees,
                "n_points_per_tree": self._n_points_per_tree,
                "ratio_features": self._ratio_features,
                "min_samples_split": self._min_samples_split,
                "min_samples_leaf": self._min_samples_leaf,
                "max_depth": self._max_depth,
                "eps_purity": self._eps_purity,
                "max_nodes": self._max_nodes,
                "bootstrapping": self._bootstrapping,
                "pca_components": self._pca_components,
            }
        )

        return meta

    def _train(self, X: np.ndarray, y: np.ndarray) -> RandomForestMO:
        X = self._impute_inactive(X)
        y = y.flatten()
        # self.X = X
        # self.y = y.flatten()

        # Code not checked
        # if self._n_points_per_tree <= 0:
        #     self._rf_opts.num_data_points_per_tree = X.shape[0]
        # else:
        #     self._rf_opts.num_data_points_per_tree = self._n_points_per_tree

        # Moved instatiation of model (XGBRegressor) in __init__
        # self._rf = regression.binary_rss_forest()
        # self._rf.options = self._rf_opts

        # data = self._init_data_container(X, y)        
        # self._rf.fit(data, rng=self._rng)
        
        # TODO: Add random state like _rng, or at least seed
        self._rf.fit(X, y)

        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._types), X.shape[1]))

        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        assert self._rf is not None
        # In case of missing configuration
        X = self._impute_inactive(X)

        _n_configs = X.shape[0]
        _n_entries = len(self._types)
        # Example: (2 configs, 5 entries) -> [[4.17 1.14 1.46 1.86 3.96] [7.20 3.02 9.23 3.45 5.38]]
        assert X.shape == (_n_configs, _n_entries), f"Shape of X is {X.shape} instead of ({_n_configs}, {_n_features}) -> (#configs + #features)"

        if self._log_y:
            raise NotImplementedError("log_y still not implemented for XGBRegressor")
        else:
            for row_X in X:
                # print(self._rf.get_xgb_params())
                # mean_, var = self._rf.predict_mean_var(row_X)
                pass

            out = self._rf.predict(X)
            """
            Example output of "self._rf.predict(X)" with shape: (3 instances/configs, 4 targets).
            Indicies are leafs. Indicies are shared between trees.
                [[ 4.560528  10.060061  26.21033   51.86771  ]
                 [ 6.300218   8.974303  43.81962   23.868626 ]
                 [ 3.8757503  4.475569   5.692934  76.91086  ]]

            Other functions that yield the same output:
                self._rf.predict(X, output_margin=True)
                booster = self._rf.get_booster()
                booster.predict(xgboost.DMatrix(X), output_margin=True)
            """

            # leaf_idx = self._rf.apply(X)
            """
            Example output of "self._rf.apply(X)" with shape: (3 instances/configs, 10 trees).
                [[1. 1. 2. 2. 1. 2. 3. 3. 3. 3.]
                 [2. 2. 2. 2. 2. 2. 4. 4. 4. 4.]
                 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]

            Other function that yield the same output:
                booster = self._rf.get_booster()
                booster.predict(xgboost.DMatrix(X), pred_leaf=True)                        
            """

            # booster = self._rf.get_booster()
            # booster.predict(xgboost.DMatrix(X), pred_contribs=True)
            # booster.get_dump()
            """
            Function that should fetch the value for each leaf:
                booster.predict(xgboost.DMatrix(X), pred_contribs=True)
            Now returning:
                failed: !model.learner_model_param->IsVectorLeaf(): Predict
                contribution support for multi-target tree is not yet implemented. 
                Raising error here: https://github.com/dmlc/xgboost/blob/6fd4a306670fa82da06b42ef217705eefd97cbcd/src/predictor/cpu_predictor.cc#L857

            Alternative function:
                booster.get_dump()
            Now returning:
                XGBoostError: Check failed: !IsMultiTarget():
                Raising error here: https://github.com/dmlc/xgboost/blob/6fd4a306670fa82da06b42ef217705eefd97cbcd/src/tree/tree_model.cc#L740
            """
            mean, var = out, None
            # mean, var = np.mean(X), np.var(X)
            # assert mean is not None and var is not None
            assert out.shape == (_n_configs, self._n_objectives), f"Shape of mean is {mean.shape} instead of {(_n_configs, self._n_objectives)} -> (#configs, #objectives)"
        return mean, var

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance marginalized over all instances.

        Note
        ----
        The method is random forest specific and follows the SMAC2 implementation. It requires
        no distribution assumption to marginalize the uncertainty estimates.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameter + #features]
            Input data points.

        Returns
        -------
        means : np.ndarray [#samples, 1]
            The predictive mean.
        vars : np.ndarray [#samples, 1]
            The predictive variance.
        """
        assert self._n_features == 0, "Removed code with self._n_features>=1, reimplement it"
        mean, var = self.predict(X)

        assert var is None, "Removed code to handle variance, reimplement it"
        # assert var is not None
        # var[var < self._var_threshold] = self._var_threshold
        # var[np.isnan(var)] = self._var_threshold

        assert mean.shape == (X.shape[0], self._n_objectives), f"Shape of mean is {mean.shape} instead of {(X.shape[0], self._n_objectives)} -> (#configs, #objectives)"
        return mean, var
