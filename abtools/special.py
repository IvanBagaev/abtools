# -*- coding: utf-8 -*-
"""LAL."""
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

from tqdm import tqdm

from .base import BaseTest
from ..plotting import CLRS_LT, CLRS_DK


__all__ = [
    'PermutationTest',
    'ZTest',
    'TTest'
]


class PermutationTest(BaseTest):
    """
    Implementation of Fisher's permutation test.

    A Fisher's exact permutation test is a type of statistical significance
    test in which the distribution of the test statistic under the
    null hypothesis is obtained by calculating all possible values of the test
    statistic under rearrangements of the labels on the observed data points.
    In other words, the method by which treatments are allocated to subjects in
    an experimental design is mirrored in the analysis of that design. If the
    labels are exchangeable under the null hypothesis, then the resulting tests
    yield exact significance levels.

    Parameters
    ----------
    a, b : {list, ndarray}
        Observed data for two groups, where a - control group,  b - test group.
    alpha : float
        Significance level.

    Returns
    -------
    p_value : float
        Two-sided p-value.
    sign : bool
        Rejected or not null hypotesis with given significance level.

    References
    ----------

    [1] - Fisher, R. A. (1935). The design of experiments. 1935.
    Oliver and Boyd, Edinburgh.

    [2] - Ernst, M. D. (2004). Permutation methods: basis for exact inference.
    Statistical Science, 19(4), 676-685
    """

    def compute_test_statistic(self, a, b):
        return self.diff

    def compute_critical(self, a, b):
        z = np.concatenate([a, b])
        n = len(a)
        null_dist = []
        # start permutations
        for j in tqdm(range(10000)):
            np.random.shuffle(z)
            null_dist.append(np.mean(z[:n]) - np.mean(z[n:]))

        self.null_dist = np.array(null_dist)
        return np.abs(np.percentile(self.null_dist, 1 - self.alpha / 2))

    def compute_p_value(self, a, b):
        return np.mean(np.abs(self.diff) < np.abs(self.null_dist))

    def compute_confidence_intervals(self, a, b):
        ci = [np.percentile(self.null_dist, p*100)
              for p in [0 + self.alpha/2, 1 - self.alpha/2]]
        return ci

    def plot(self, plt_kwds={}, hist_kwds={}):
        """Plot null distribution of permutation test."""
        fig = plt.figure(**plt_kwds)
        plt.hist(
            self.null_dist,
            50,
            histtype='step',
            color=CLRS_LT[1],
            label='Null distribution',
            **hist_kwds
        )

        plt.legend()
        ci_lbl = ('%d%% confidence interval' % ((1-self.alpha) * 100))
        plt.axvline(self.diff, label='Observed difference')
        plt.axvline(self.confidence_intervals[0], linestyle='--')
        plt.axvline(self.confidence_intervals[1], linestyle='--', label=ci_lbl)

        plt.title("Fisher's permutation test")
        plt.legend(bbox_to_anchor=(1.05, .65))