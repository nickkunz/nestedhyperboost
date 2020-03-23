## load libraries
import random as rd
from xgboost import XGBClassifier
from nestedhyperboost.argument_quality import ArgumentQuality
from nestedhyperboost.xgboost.xgb_params import xgb_params
from nestedhyperboost.ncv_optimizer import ncv_optimizer

## xgboost classification
def xgb_ncv_classifier(
    
    data,          ## pandas dataframe, clean (no nan's)
    y,             ## string, header of y reponse variable
    loss = None,   ## string, objective function to minimize
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
    
    ## initiate modeling method
    method = XGBClassifier
    params = xgb_params()
    
    ## initiate prediction type
    num_uni_val = len(data[y].unique())
    
    if num_uni_val > 2:
        pred_type = "multi-class"
        params["num_class"] = len(data[y].unique())
        
        if loss is None:
            loss = "multi:softmax"
    
    if num_uni_val is 2:
        pred_type = "binary"
        
        if loss is None:
            loss = "reg:logistic"
    
    if num_uni_val == 1:
        print("y response variable values are constant")
    
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
    
    ## classification results object
    return ncv_results