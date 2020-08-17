import numpy as np
import scipy as sp
import scipy.stats

from numpy import sqrt


class HypothesisTest(object):
    
    def __init__(self, a, b, alpha=0.05):
        a, b = np.array(a), np.array(b)
        self.name = 'Base Test'
        self.alpha = alpha
        self.diff = b.mean() - a.mean()
        self.statistic = self.test_statistic(a, b)
        self.critical = self.critical(a, b)
        self.p_value = self.p_value(a, b)

        self.sign = self.p_value <= self.alpha
        self.confidence_intervals = self.confidence_intervals(a, b)

        if self.statistic > self.critical:
            self.result = 'E(A) < E(B)'
        elif self.statistic < -self.critical:
            self.result = 'E(A) > E(B)'
        else:
            self.result = 'E(A) = E(B)'

    def test_statistic(self, a, b):
        raise NotImplementedError

    def critical(self, a, b):
        raise NotImplementedError

    def p_value(self, a, b):
        raise NotImplementedError

    def confidence_intervals(self, a, b):
        raise NotImplementedError

    def summary(self):
        """Print test summary."""
        print("Observed difference of means (E(B) - E(A)) = %.4f"
              % self.diff)
        print("Test statistic is %.4f and critical value is %.4f"
              % (self.statistic, self.critical))

        print("Two-sided p-value = %.2f" % self.p_value)

        print("Null hypotesis is %s with %.2f significance level"
              % (('not ' * (1 * (not self.sign)) + 'rejected'), self.alpha))

        print(self.result)

        
class ZTest(HypothesisTest):

    def ci(self, x):
        x = np.array(x)
        m, se = np.mean(x), sp.stats.sem(x)
        h = se * sp.stats.norm.ppf(1 - self.alpha / 2)
        return m - h, m + h

    def test_statistic(self, a, b):
        a_mean = a.mean()
        avar = a.var()
        na = a.size

        b_mean = b.mean()
        bvar = b.var()
        nb = b.size

        z = ((b_mean - a_mean)) / np.sqrt(avar/na + bvar/nb)
        return z

    def critical(self, a, b):
        return sp.stats.norm.ppf(1 - self.alpha / 2)

    def p_value(self, a, b):
        return 2 * (1 - sp.stats.norm.cdf(abs(self.statistic)))

    def confidence_intervals(self, a, b):
        self.significance = max(2*sp.stats.norm.cdf(abs(a.mean() - b.mean()) /
                                (sp.stats.sem(a) + sp.stats.sem(b))) - 1, 0)
        return self.ci(a), self.ci(b)

class UTest(HypothesisTest):

    def p_value(self, a, b):
        _, u_test_p_value = stats.mannwhitneyu(variant_a, variant_b)
        return u_test_p_value


class Model():
    """Base model for A/B test analysis"""
    def __init__(self, x):
        raise NotImplementedError()

    def rvs(self, random_size=100000):
        return self._rvs(random_size)
    
    def _rvs(self, random_size):
        raise NotImplementedError()

        
class BernoulliModel(Model):
    """Model for binary data"""
    def __init__(self, x):
        self.alpha = x.sum() + 0.5
        self.beta = len(x) - self.alpha + 0.5
    
    def _rvs(self, random_size):
        return scipy.stats.beta.rvs(self.alpha, self.beta, size=random_size)

    
class NormalModel(Model):
    """Bayesian Gaussian model for normal distributed metrics"""
    def __init__(self, x):
        self.mu = x.mean()
        self.sigma = x.std() / sqrt(len(x))
    
    def _rvs(self, random_size):
        return scipy.stats.norm.rvs(self.mu, self.sigma, size=random_size)

    
class LognormalModel(Model):
    """Bayesian Lognormal model for skewed data."""
    def __init__(self, x):
        log_x = np.log(x)
        std = log_x.std()
        n = len(log_x)
        self.mu__mu = log_x.mean()
        self.mu__sigma = std / sqrt(n)
        self.var__alpha = (n - 1) / 2
        self.var__beta = self.var__alpha * std**2
    def _rvs(self, random_size):
        mu = scipy.stats.norm.rvs(self.mu__mu, self.mu__sigma, size=random_size)
        var = scipy.stats.invgamma.rvs(self.var__alpha, scale=self.var__beta, size=random_size)
        return np.exp(mu + var / 2)

    
class ARPUModel(Model):
    """Bayesian model for testing skewed data with large amount of zeros"""
    def __init__(self, x):
        # binary part modeled with bernoulli
        self.conversion_model = BernoulliModel((x > 0) * 1)
        # countinious part modeled with Lognormal
        self.ARPPU_model = LognormalModel(x[x > 0])
    def _rvs(self, random_size):
        return self.conversion_model.rvs(random_size) * self.ARPPU_model.rvs(random_size)

    
class StatTest:
    """Base statistical test"""
    def __init__(self, a, b):
        raise NotImplementedError()

    def test(self, alpha=0.05, tail='both', a_name='E(A)', b_name='E(B)'):
        p = self.probability()
        p_critical = self.critical(alpha, tail)
        if p > p_critical[1]:
            return ' < '.join([a_name, b_name])
        elif p < p_critical[0]:
            return ' > '.join([a_name, b_name])
        else:
            return ' = '.join([a_name, b_name])

    def critical(self, alpha=0.05, tail='both'):
        if tail == 'left':
            return [alpha, 1]
        elif tail == 'right':
            return [0, 1 - alpha]
        else:
            return [alpha / 2, 1 - alpha / 2]

    def probability(self):
        return self._probability()

    def _probability(self):
        raise NotImplementedError()

    
class BTest(StatTest):
    def __init__(self, a, b, model_name, random_size=100000):
        self.diff = model_name(b).rvs(random_size) - model_name(a).rvs(random_size)

    def _probability(self):
        return (self.diff > 0).mean()
