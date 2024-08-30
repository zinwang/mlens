"""ML-ENSEMBLE

Test classes.
"""

import numpy as np
from mlens.index import FullIndex, FoldIndex
from mlens.testing import Data
from mlens.testing.dummy import ESTIMATORS, PREPROCESSING
from mlens.utils.dummy import OLS, Scale
from mlens.utils.exceptions import NotFittedError, ParameterChangeWarning
from mlens.parallel import make_group
from mlens.estimators import (
    LearnerEstimator,
    TransformerEstimator,
    LayerEnsemble,
)
from mlens.externals.sklearn.base import clone

try:
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.utils.validation import check_X_y, check_array

    run_sklearn = True
except ImportError:
    check_estimator = None
    run_sklearn = False

data = Data("stack", False, True)
X, y = data.get_data((25, 4), 3)
(F, wf), (P, wp) = data.ground_truth(X, y)

Est = LayerEnsemble
est = LayerEnsemble(
    make_group(FoldIndex(), ESTIMATORS, PREPROCESSING), dtype=np.float64
)


class Tmp(Est):
    """Temporary class

    Wrapper to get full estimator on no-args instantiation. For compatibility
    with older Scikit-learn versions.
    """

    def __init__(self):
        args = {
            LearnerEstimator: (OLS(), FullIndex()),
            LayerEnsemble: (
                make_group(FullIndex(), ESTIMATORS, PREPROCESSING),
            ),
            TransformerEstimator: (Scale(), FullIndex()),
        }[Est]
        super(Tmp, self).__init__(*args)

    __init__.deprecated_original = __init__

    def fit(self, X, y, *args, **kwargs):
        X, y = check_X_y(X, y)
        return super(Tmp, self).fit(X, y, *args, **kwargs)

    def fit_transform(self, X, y, *args, **kwargs):
        X, y = check_X_y(X, y)
        return super(Tmp, self).fit_transform(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        X = check_array(X)
        return super(Tmp, self).predict(X, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        X = check_array(X)
        return super(Tmp, self).transform(X, *args, **kwargs)


# These are simple run tests to ensure parallel wrapper register backend.
# See parallel for more rigorous tests


def test_layer_fit():
    """[Module | LayerEstimator] test fit"""
    out = est.fit(X, y)
    assert out is est

    p = est.fit_transform(X, y, refit=False)
    np.testing.assert_array_equal(p, F)


def test_layer_transform():
    """[Module | LayerEnsemble] test transform"""
    p = est.transform(X)
    np.testing.assert_array_equal(p, F)


def test_layer_predict():
    """[Module | LayerEnsemble] test predict"""
    p = est.predict(X)
    np.testing.assert_array_equal(p, P)


def test_layer_clone():
    """[Module | LayerEnsemble] test clone"""
    cl = clone(est)
    p = cl.fit_transform(X, y)
    np.testing.assert_array_equal(p, F)


def test_layer_params_estimator():
    """[Module | LayerEnsemble] test set params on estimator"""
    est.fit(X, y)

    # Just a check that this works
    out = est.get_params()
    assert isinstance(out, dict)

    est.set_params(**{"offs-1__estimator__offset": 10})
    np.testing.assert_warns(ParameterChangeWarning, est.predict, X)


def test_layer_params_indexer():
    """[Module | LayerEnsemble] test set params on indexer"""
    est.fit(X, y)

    est.set_params(**{"null-1__indexer__folds": 3})
    np.testing.assert_warns(ParameterChangeWarning, est.predict, X)


def test_layer_attr():
    """[Module | LayerEnsemble] test setting attribute"""

    def fitted():
        est.__fitted__

    est.propagate_features = [0]
    np.testing.assert_warns(ParameterChangeWarning, fitted)

    # If this fails, it is trying to propagate feature but predict_out is None!
    est.fit(X, y)
    np.testing.assert_no_warnings(fitted)


if run_sklearn:

    def test_layer():
        """[Module | LayerEnsemble] test pass estimator checks"""
        check_estimator(Tmp)
