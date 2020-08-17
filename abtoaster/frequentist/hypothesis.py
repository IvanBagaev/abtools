
# https://www.mathbootcamps.com/calculating-confidence-intervals-for-the-mean/
# https://mc.ai/how-to-code-the-students-t-test-from-scratch-in-python/


class HypothesisTest(object):
    
    def __init__(self, a, b, alpha=0.05):
        a, b = np.array(a), np.array(b)
        self.name = 'Base Test'
        self.alpha = alpha

        self.ate = b_mean - a_mean

        self.sign = self.p_value <= self.alpha
        self.confidence_intervals = self.confidence_intervals(a, b)

        if self.statistic > self.critical:
            self.result = 'E(A) < E(B)'
        elif self.statistic < -self.critical:
            self.result = 'E(A) > E(B)'
        else:
            self.result = 'E(A) = E(B)'

    def fit(self, a, b):
        self.a_mean = np.mean(a)
        self.b_mean = np.mean(b)
        self.sem_a = sp.stats.sem(a)
        self.sem_b = sp.stats.sem(b) 

    @property
    def test_statistic(self):
        return self._test_statistic()

    @property
    def critical(self):
        return self._critical()

    @property
    def p_value(self):
        return self._p_value()

    @property
    def confidence_intervals(self):
        raise NotImplementedError

    def summary(self):
        """Print test summary."""
        print("Observed Average Treatment Effect (E(B) - E(A)) = %.4f"
              % self.ate)
        print("Test statistic is %.4f and critical value is %.4f"
              % (self.statistic, self.critical))

        print("Two-sided p-value = %.2f" % self.p_value)

        print("Null hypotesis is %s with %.2f significance level"
              % (('not ' * (1 * (not self.sign)) + 'rejected'), self.alpha))

        print(self.result)

    def _test_statistic(self):
        raise NotImplementedError

    def _critical(self):
        raise NotImplementedError

    def _p_value(self):
        raise NotImplementedError

    def _confidence_intervals(self):
        raise NotImplementedError

        
class ParametricTest(HypothesisTest):
    pass

class NonParametricTest(HypothesisTest):
    pass

class ZTest(ParametricTest):

    def _ci(self, x):
        x = np.array(x)
        m, se = np.mean(x), sp.stats.sem(x)
        h = se * sp.stats.norm.ppf(1 - self.alpha / 2)
        return m - h, m + h

    def _test_statistic(self):
        a_mean = a.mean()
        avar = a.var()
        na = a.size

        b_mean = b.mean()
        bvar = b.var()
        nb = b.size

        z = ((b_mean - a_mean)) / np.sqrt(avar/na + bvar/nb)
        return z

    def _critical(self):
        return sp.stats.norm.ppf(1 - self.alpha / 2)

    def _p_value(self):
        return 2 * (1 - sp.stats.norm.cdf(abs(self.test_statistic)))

    def _confidence_intervals(self):
        self.significance = max(2*sp.stats.norm.cdf(abs(self.a_mean - self.b_mean) /
                                 (self.sem_a + self.sem_b)) - 1, 0)
                                #(sp.stats.sem(a) + sp.stats.sem(b))) - 1, 0)
        return self._ci(a), self._ci(b)

class TTest(ParametricTest):
    pass

class UTest(HypothesisTest):

    def p_value(self, a, b):
        _, u_test_p_value = stats.mannwhitneyu(a, b)
        return u_test_p_value
