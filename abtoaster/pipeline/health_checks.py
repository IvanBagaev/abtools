from .data import ABDataset

# TODO: 

class HealthCheck:
    def __init__(self, dataset: ABDataset):
        self.dataset = dataset

    def check(self, *args, **kwargs) -> bool:
        return self._check(*args, **kwargs)

    def _check(self, *args, **kwargs):
        return NotImplementedError

class SampleSizeImbalance(HealthCheck):
    """
    Health Check for Sample size imbalance between control and treatment groups
    """
    def _check(self, *args, **kwargs):
        result, is_healthy =  {'message': 'This check is not implemented! Placeholder message'}, True
        return result, is_healthy

class Flickers(HealthCheck):
    """
    Health Check for users who changed test variant being in experiment.
    """
    def _check(self, *args, **kwargs):
        result, is_healthy =  {'message': 'This check is not implemented! Placeholder message'}, True
        return result, is_healthy