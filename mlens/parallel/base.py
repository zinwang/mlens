"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Base classes for parallel estimation


Schedulers for global setups:
   0:
      Base setups - independent of other features:
         IndexMixin._setup_0_index

   1:
      Global setups - reserved for aggregating classes:
         Layer._setup_1_global

   2:
      Dependents on 0:
         ProbaMixin.__setup_2_multiplier

   3:
      Dependents on 0, 2:
         OutputMixin.__setup_3__output_columns

Note that schedulers are experimental and may change without a deprecation
cycle.
"""

import warnings
from abc import abstractmethod
import numpy as np

from ._base_functions import check_stack, check_params
from .. import config
from ..utils.exceptions import ParallelProcessingError
from ..externals.sklearn.base import clone, BaseEstimator as _BaseEstimator


class ParamMixin(_BaseEstimator, object):
    """Parameter Mixin

    Mixin for protecting static parameters from changes after fitting.

    .. Note::
       To use this mixin the instance inheriting it must set
       ``__static__=list()`` and ``_static_fit_params_=dict()``
       in ``__init__``.
    """

    def _store_static_params(self):
        """Record current static params for future comparison."""
        if self.__static__:
            for key, val in self.get_params(deep=False).items():
                if key in self.__static__:
                    self._static_fit_params[key] = clone(val, safe=False)

    def _check_static_params(self):
        """Check if current static params are identical to previous params"""
        current_static_params = {
            k: v
            for k, v in self.get_params(deep=False).items()
            if k in self.__static__
        }
        return check_params(self._static_fit_params, current_static_params)


class IndexMixin(object):
    """Indexer mixin

    Mixin for handling indexers.

    .. note::
       To use this mixin the instance inheriting it must set the
        ``indexer`` or ``indexers`` attribute in ``__init__`` (not both).
    """

    @property
    def __indexer__(self):
        """Flag for existence of indexer"""
        return hasattr(self, "indexer") or hasattr(self, "indexers")

    def _check_indexer(self, indexer):
        """Check consistent indexer classes"""
        cls = indexer.__class__.__name__.lower()
        if "index" not in cls:
            ValueError("Passed indexer does not appear to be valid indexer")

        lcls = [idx.__class__.__name__.lower() for idx in self._get_indexers()]
        if lcls:
            if "blendindex" in lcls and cls != "blendindex":
                raise ValueError(
                    "Instance has blendindex, but was passed full type"
                )
            elif "blendindex" not in lcls and cls == "blendindex":
                raise ValueError(
                    "Instance has full type index, but was passed blendindex"
                )

    def _get_indexers(self):
        """Return list of indexers"""
        if not self.__indexer__:
            raise AttributeError("No indexer or indexers attribute available")
        indexers = [getattr(self, "indexer", None)]
        if None in indexers:
            indexers = getattr(self, "indexers", [None])
        return indexers

    def _setup_0_index(self, X, y, job):
        indexers = self._get_indexers()
        for indexer in indexers:
            indexer.fit(X, y, job)


class OutputMixin(IndexMixin):
    """Output Mixin

    Mixin class for interfacing with ParallelProcessing when outputs are
    desired.

    .. note::
       To use this mixin the instance inheriting it must set the
       ``feature_span`` attribute and ``__no_output__`` flag in ``__init__``.
    """

    @abstractmethod
    def set_output_columns(self, X, y, job, n_left_concats=0):
        """Set output columns for prediction array"""
        pass

    def _setup_3_output_columns(self, X, y, job, n_left_concats=0):
        """Set output columns for prediction array. Used during setup"""
        if not self.__no_output__:
            self.set_output_columns(X, y, job, n_left_concats)

    def shape(self, job):
        """Prediction array shape"""
        if not hasattr(self, "feature_span"):
            raise ParallelProcessingError(
                "Instance dose not set the feature_span attribute "
                "in the constructor."
            )

        if not self.feature_span:
            raise ValueError("Columns not set. Call set_output_columns.")
        return self.size(job), self.feature_span[1]

    def size(self, attr):
        """Get size of dim 0"""
        if attr not in ["n_test_samples", "n_samples"]:
            attr = "n_test_samples" if attr != "predict" else "n_samples"

        indexers = self._get_indexers()
        sizes = list()
        for indexer in indexers:
            sizes.append(getattr(indexer, attr))

        sizes = np.unique(sizes)
        if not sizes.shape[0] == 1:
            warnings.warn(
                "Inconsistent output sizes generated by indexers "
                "(sizes: %r from indexers %r).\n"
                "outputs will be zero-padded" % (sizes.tolist(), indexers)
            )
            return max(sizes)
        return sizes[0]


class ProbaMixin(object):
    """ "Probability Mixin

    Mixin for probability features on objects
    interfacing with :class:`~mlens.parallel.backend.ParallelProcessing`

    .. note::
       To use this mixin the instance inheriting it must set the ``proba``
       and the ``_classes(=None)``attribute in ``__init__``.
    """

    def _setup_2_multiplier(self, X, y, job=None):
        if self.proba and y is not None:
            self.classes_ = y

    def _get_multiplier(self, X, y, alt=1):
        if self.proba:
            multiplier = self.classes_
        else:
            multiplier = alt
        return multiplier

    @property
    def _predict_attr(self):
        return "predict" if not self.proba else "predict_proba"

    @property
    def classes_(self):
        """Prediction classes during proba"""
        return self._classes

    @classes_.setter
    def classes_(self, y):
        """Set classes given input y"""
        self._classes = np.unique(y).shape[0]


class BaseBackend(object):
    """Base class for parallel backend

    Implements default backend settings.
    """

    def __init__(
        self, backend=None, n_jobs=-1, dtype=None, raise_on_exception=True
    ):
        self.n_jobs = n_jobs
        self.dtype = dtype if dtype is not None else config.get_dtype()
        self.backend = backend if backend is not None else config.get_backend()
        self.raise_on_exception = raise_on_exception

    @abstractmethod
    def __iter__(self):
        yield


class BaseParallel(BaseBackend):
    """Base class for parallel objects

    Parameters
    ----------
    name : str
        name of instance. Should be unique.

    backend : str or object (default = 'threading')
        backend infrastructure to use during call to
        :class:`mlens.externals.joblib.Parallel`. See Joblib for further
        documentation. To set global backend,
        see :func:`~mlens.config.set_backend`.

    raise_on_exception : bool (default = True)
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer. If set to ``False``, warnings are issued instead
        but estimation continues unless exception is fatal. Note that this
        can result in unexpected behavior unless the exception is anticipated.

    verbose : int or bool (default = False)
        level of verbosity.

    n_jobs : int (default = -1)
        Degree of concurrency in estimation. Set to -1 to maximize,
        1 runs on a single process (or thread).

    dtype : obj (default = np.float32)
        data type to use, must be compatible with a numpy array dtype.
    """

    def __init__(self, name, *args, **kwargs):
        super(BaseParallel, self).__init__(*args, **kwargs)
        self.name = name
        self.__no_output__ = False

    @abstractmethod
    def __iter__(self):
        """Iterator for process manager"""
        yield

    def setup(self, X, y, job, skip=None, **kwargs):
        """Setup instance for estimation"""
        skip = ["_setup_%s" % s for s in skip] if skip else []
        funs = [
            f for f in dir(self) if f.startswith("_setup_") and f not in skip
        ]

        for f in sorted(funs):
            func = getattr(self, f)
            args = func.__func__.__code__.co_varnames
            fargs = {k: v for k, v in kwargs.items() if k in args}
            func(X, y, job, **fargs)


class BaseEstimator(ParamMixin, _BaseEstimator, BaseParallel):
    """Base Parallel Estimator class

    Modified Scikit-learn class to handle backend params that we want to
    protect from changes.
    """

    def __init__(self, *args, **kwargs):
        super(BaseEstimator, self).__init__(*args, **kwargs)
        self.__static__ = list()
        self._static_fit_params = dict()

    def get_params(self, deep=True):
        out = super(BaseEstimator, self).get_params(deep=deep)
        for name in BaseBackend.__init__.__code__.co_varnames:
            if name not in ["self"]:
                out[name] = getattr(self, name)
        return out

    @property
    @abstractmethod
    def __fitted__(self):
        """Fit status"""
        return self._check_static_params()


class BaseStacker(BaseEstimator):
    """Base class for instanes that stack job estimators"""

    def __init__(self, stack=None, verbose=False, *args, **kwargs):
        super(BaseStacker, self).__init__(*args, **kwargs)
        if stack and not isinstance(stack, list):
            raise ValueError("Stack must be a list. Got %r:" % type(stack))
        self.stack = stack if stack else list()
        self._verbose = verbose

    @abstractmethod
    def __iter__(self):
        yield

    def push(self, *stack):
        """Push onto stack"""
        check_stack(stack, self.stack)
        for item in stack:
            self.stack.append(item)
            attr = item.name.replace("-", "_").replace(" ", "").strip()
            setattr(self, attr, item)
        return self

    def replace(self, idx, item):
        """Replace a current member of the stack with a new instance"""
        attr = item.name.replace("-", "_").replace(" ", "").strip()
        setattr(self, attr, item)
        self.stack[idx] = item

    def pop(self, idx):
        """Pop a previous push with index idx"""
        return self.stack.pop(idx)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            whether to return nested parameters.
        """
        out = super(BaseStacker, self).get_params(deep=deep)
        if not deep:
            return out

        for item in self.stack:
            out[item.name] = item
            for key, val in item.get_params(deep=True).items():
                out["%s__%s" % (item.name, key)] = val
        return out

    @property
    def __fitted__(self):
        """Fitted status"""
        if not self.stack or not self._check_static_params():
            return False
        return all([g.__fitted__ for g in self.stack])

    @property
    def __stack__(self):
        """Check stack"""
        if not isinstance(self.stack, list):
            raise ValueError(
                "Stack corrupted. Extected list. Got %r" % type(self.stack)
            )
        return len(self.stack) > 0

    @property
    def verbose(self):
        """Verbosity"""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """Set verbosity"""
        self._verbose = verbose
        for g in self.stack:
            g.verbose = verbose
