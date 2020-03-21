## load libraries
import pandas as pd

## create input quality checks for regressor and classifier
class ArgumentQuality():
    def __init__(self, data, y, loss, k_outer, k_inner, n_evals, seed, verbose):
        
        self.data = data
        self.y = y
        self.loss = loss
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.n_evals = n_evals
        self.seed = seed
        self.verbose = verbose
        
        ## quality check for dataframe
        if type(self.data) is not pd.DataFrame:
            print("must pass pandas dataframe into 'data' argument")
        
        ## quality check for missing values in dataframe
        if self.data.isnull().values.any():
            print("dataframe cannot contain missing values")
        
        ## quality check for y
        if isinstance(self.y, str) is False:
            print("'y' must be a string")
        
        if self.y in self.data.columns.values is False:
            print("'y' must be an header name (string) found in the dataframe")
        
        ## quality check for loss 
        if isinstance(self.loss, str) is False:
            print("'loss' must be a string")
        
        ## quality check for k-fold outer argument
        if self.k_outer > len(self.data):
            print("'k_outer' is greater than number of observations (rows)")
        
        if self.k_outer < 2:
            print("'k_outer' must be a positive integer greater than 1")
        
        ## quality check for k-fold inner argument
        if self.k_inner > len(self.data):
            print("'k_inner' is greater than number of observations (rows)")
        
        if self.k_inner < 2:
            print("'k_inner' must be a positive integer greater than 1")
        
        ## quality check for number of evaluations
        if self.n_evals < 1:
            print("'n_evals' must be a positive integer")
        
        ## quality check for random seed
        if self.seed < 1:
            print("'seed 'must be a positive integer")
        
        ## quality check for verbose
        if type(self.verbose) is not bool:
            print("'verbose' must be boolean")

## create input quality checks for optimizer
class ArgumentQualityOptimizer(ArgumentQuality):
    def __init__(self, data, y, loss, k_outer, k_inner, n_evals, seed, verbose,
                 pred_type, method, params):
        super().__init__(data, y, loss, k_outer, k_inner, n_evals, seed, verbose)
        
        self.pred_type = pred_type
        self.method = method
        self.params = params
        
        ## quality check for pred
        if self.pred_type in ["regress", "multi-class", "binary"] is False:
            print("'pred' must be 'regress' or 'multi-class' or 'binary'")
        
        ## quality check for dataframe
        if type(self.params) is not dict:
            print("'params' must be a dictionary")
