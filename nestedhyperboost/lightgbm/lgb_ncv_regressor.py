## load libraries
import random as rd
from lightgbm import LGBMRegressor
from nestedhyperboost.argument_quality import ArgumentQuality
from nestedhyperboost.lightgbm.lgb_params import lgb_params
from nestedhyperboost.ncv_optimizer import ncv_optimizer

## lightgbm regression
def lgb_ncv_regressor(
    
    data,          ## pandas dataframe, clean (no nan's)
    y,             ## string, header of y reponse variable
    loss = "rmse", ## string, objective function to minimize
    k_outer = 5,   ## pos int, k number of outer folds (1 < k < n)
    k_inner = 5,   ## pos int, k number of inner folds (1 < k < n)
    n_evals = 25,  ## pos int, number of evals for bayesian optimization
    seed = rd.randint(0, 9999),  ## pos int, fix for reproduction
    verbose = True               ## bool, display output
    ):
    
    ## conduct input quality checks
    ArgumentQuality(
        data = data,
        y = y,
        loss = loss,
        k_outer = k_outer,
        k_inner = k_inner,
        n_evals = n_evals,
        seed = seed,
        verbose = verbose
    )
    
    ## initiate modeling method and prediction type
    pred_type = "regress"
    method = LGBMRegressor
    params = lgb_params()
    
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
