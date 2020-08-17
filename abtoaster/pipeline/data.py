from pandas import DataFrame, Series

# from .health_checks import Flickers, SampleSizeImbalance


class ABDataset:
    """
    Class for storing A/B/N test data ready for pipeline.

    Dataset defined as following:
    - `X` pandas DataFrame with user's attributes
    - w pandas Series with A/B test variant assignments
    - y pandas Series for Target Metric
    """
    def __init__(self, X: DataFrame, y: Series, w: Series, control_keys=None, test_entry_key='join_date', verbose=True):
        # detecting variants
        variants = w.unique()
        if control_keys is None:
            control_keys = [x for x in self.variants 
                if 'control' in x.lower() or 'holdout' in x.lower()]
        else:
            # TODO: check types
            control_keys = control_keys
        # A/B test data
        ## features
        self.X = X
        ## variant assignments
        self.w = w
        ## target metric
        self.y = y
        self.control_keys = control_keys
        self.variants = variants
        self.test_entry_key = test_entry_key
        self.verbose = verbose

    def health_check(self):
        """
        Performes A/B test health check for sample size imbalance and flickers.
        
        - **SampleSizeImbalanceCheck** - when variants have at least 5% difference between sample sizes
        - **FlickersCheck**  - check for users who changed the variant during the experiment 
        """
        # TODO: verbose
        checks = {}
        results = []
        for check_type in (Flickers, SampleSizeImbalance):
            check = check_type(self)
            result, is_healthy = check.check()
            results.append(is_healthy)
            checks.update(result)
            if self.verbose:
                print(result)
        self.checks = checks
        return all(results)
        

