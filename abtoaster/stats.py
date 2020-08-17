import numpy as np

from scipy.stats import ks_2samp, entropy


def histogram_intersection(a, b, bins=256) -> float:
    """Calculate percentage of two arrays' histograms intersection"""
    minval = min(a.min(), b.min())
    maxval = max(a.max(), b.max())
    h1, bins= np.histogram(a, bins=bins, range=(minval, maxval), normed=True)
    h2, _ = np.histogram(b, bins=bins, range=(minval, maxval), normed=True)
    bins = np.diff(bins)
    sm = 0
    for i in range(len(bins)):
        sm += min(bins[i]*h1[i], bins[i]*h2[i])
    return sm

def ks(a, b) -> float:
    """Calculate D-value (test statistic) of Kolmogorov-Smirnov test"""
    dvalue, _ = ks_2samp(a, b)
    return dvalue