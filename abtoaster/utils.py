import pandas as pd
import numpy as np

from typing import Callable


def estimator(func: Callable):
    """
    Decorator for estimator functions parameter checking
    
    :func: is callable function of two variables with accept arrays and return array
    """
    def wrapper(a, b, *args, **kwargs):
        check_x = lambda x: any([isinstance(x, tp) for tp in [np.ndarray, pd.Series, list]])
        if all([check_x(var) for var in (a, b)]):
            a, b = np.array(a), np.array(b)
        else:
            raise ValueError('Parameters should be array-like: numpy, pandas series or python list!')
        if a.shape[0] == b.shape[0]:
            return func(a, b)
        else:
            raise ValueError('Arrays should have the same length.')
    return wrapper