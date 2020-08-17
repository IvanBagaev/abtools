import numpy as np

from scipy.stats import lognorm, norm


class OutlierDetector:
    """Base class for univariate outlier detection"""
    def __init__(self, log_transform=False, q=0.001, right=True, left=False, *args, **kwargs):
        self.log_transform=log_transform
        self.q = q
        self.right = right
        self.left = left

    def fit(self, X):
        return self._fit(X)

    def predict(self, X):
        return self._predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def _fit(self, X):
        raise NotImplementedError

    def _predict(self, X):
        _X = np.log(X) if self.log_transform else X
        return (_X > self.right_cut) | (_X < self.left_cut)


class ZScore(OutlierDetector):
    def _fit(self, X):
        self.right_cut = norm.ppf(1-self.q)
        self.left_cut = norm.ppf(self.q)
        _X = np.log(X) if self.log_transform else X
        self.var_mean = np.mean(_X)
        self.var_sd = np.std(_X)

    def _predict(self, X):
        _X = np.log(X) if self.log_transform else X
        z_scores = [(x - self.var_mean) / self.var_sd for x in _X]
        return (z_scores > self.right_cut) | (z_scores < self.left_cut)

class IQR(OutlierDetector):

    def _fit(self, X):
        _X = np.log(X) if self.log_transform else X
        q1, q3 = np.percentile(X, [25, 75])
        iqr = q3 - q1
        self.left_cut = q1 - iqr * 1.5
        self.right_cut = q3 + iqr * 1.5

class Residuals(OutlierDetector):
    pass

class RHO(OutlierDetector):
    """
    Supports only norm and lognorm distributions
    
    """
    def _fit(self, x):
        """Fit data to distribution function"""
        if not self.log_transform:
            self.dist = norm
        else:
            self.dist = lognorm

        self.params = self.dist.fit(x)
        self.fit_dist = self.dist(*self.params)

        alpha = self.q#  / len(x) if type(self.q) is int else self.q

        if self.right:
            self.right_cut = self.fit_dist.ppf(1-alpha)
        else:
            self.right_cut = x.max()
        
        if self.left:
            self.left_cut = self.fit_dist.ppf(alpha)
        else:
            self.left_cut = x.min()

    def _predict(self, X):
        return (X > self.right_cut) | (X < self.left_cut)