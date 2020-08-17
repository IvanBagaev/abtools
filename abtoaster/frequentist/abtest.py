# -*- coding: utf-8 -*-

from itertools import combinations
from tqdm import tqdm

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from .base import StatTest

from .hypothesis import Parametric, NonParametric


class PermutationTest(NonParametric):
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

    def __init__(self):
        raise NotImplementedError

    def test_statistic(self, a, b):
        return self.diff

    def critical(self, a, b):
        z = np.concatenate([a, b])
        n = len(a)
        null_dist = []
        # start permutations
        for j in tqdm(range(10000)):
            np.random.shuffle(z)
            null_dist.append(np.mean(z[:n]) - np.mean(z[n:]))

        self.null_dist = np.array(null_dist)
        return np.abs(np.percentile(self.null_dist, 1 - self.alpha / 2))

    def p_value(self, a, b):
        return np.mean(np.abs(self.diff) < np.abs(self.null_dist))

    def confidence_intervals(self, a, b):
        ci = [np.percentile(self.null_dist, p*100)
              for p in [0 + self.alpha/2, 1 - self.alpha/2]]
        return ci
