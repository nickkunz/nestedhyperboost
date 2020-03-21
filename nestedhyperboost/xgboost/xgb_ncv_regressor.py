## load libraries
import random as rd
from xgboost import XGBRegressor
from nestedhyperboost.argument_quality import ArgumentQuality
from nestedhyperboost.xgboost.xgb_params import xgb_params
from nestedhyperboost.ncv_optimizer import ncv_optimizer

## xgboost regression
def xgb_ncv_regressor(
    
    data,          ## pandas dataframe, clean (no nan's)
    y,             ## string, header of y reponse variable
    loss = "reg:squarederror", ## string, objective function to minimize
    k_outer = 5,   ## pos int, k number of outer folds (1 < k < n)
    k_inner = 5,   ## pos int, k number of inner folds (1 < k < n)
    n_evals = 25,  ## pos int, number of evals for bayesian optimization
    seed = rd.randint(0, 9999),  ## pos int, fix for reproduction
    verbose = True               ## bool, display output
    ):
    
    ## conduct input quality checks
    qual_check = ArgumentQuality(
        data = data, 
        y = y,
        loss = loss,
        k_outer = k_outer,
        k_inner = k_inner, 
        n_evals = n_evals, 
        seed = seed,
        verbose = verbose
    )
    
    ## return checked arguments
    data = qual_check.data
    y = qual_check.y
    loss = qual_check.loss
    k_outer = qual_check.k_outer
    k_inner = qual_check.k_inner
    n_evals = qual_check.n_evals
    seed = qual_check.seed
    verbose = qual_check.verbose
    
    ## initiate modeling method and prediction type
    pred_type = "regress"
    method = XGBRegressor
    params = xgb_params()
    
    ## nested cross-valid bayesian hyper-param optimization
    ncv_results = ncv_optimizer(
        
        ## main func args
        data = data, 
        y = y,
        loss = loss,
        k_outer = k_outer,
        k_inner = k_inner,
        n_evals = n_evals,
        seed = seed, 
        verbose = verbose, 
        
        ## pred func args
        pred_type = pred_type,
        method = method,
        params = params
    )
    
    ## regression results object
    return ncv_results
