## load libraries
import numpy as np
import pandas as pd
import random as rd
import warnings as wn

## mested k-fold cross-validation
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

## bayesian hyper-parameter optimization and modeling
from hyperopt import fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

## performance evaluation
from sklearn.metrics import precision_recall_fscore_support as prfs_score
from sklearn.metrics import mean_squared_error as rmse_score
from sklearn.metrics import accuracy_score as accu_score
from sklearn.metrics import roc_curve, roc_auc_score

## internal
from nestedhyperboost.argument_quality import ArgumentQualityOptimizer
from nestedhyperboost.results import RegressResults
from nestedhyperboost.results import MultiClassResults 
from nestedhyperboost.results import BinaryClassResults
from nestedhyperboost.method_select import method_select

## nested cross-validation and bayesian hyper-param optimization
def ncv_optimizer(
    
    ## main func args
    data, y, loss, k_outer, k_inner, n_evals, seed, verbose,
    
    ## pred func args
    pred_type, method, params
    ):
    
    """ 
    main underlying function, designed for rapid prototyping, quickly obtain 
    prediction results by compromising implementation details and flexibility
    
    can be applied to regression, multi-class classification, and binary 
    classification problems, unifies three important supervised learning 
    techniques for structured data:
    
    1) nested k-fold cross validation (minimize bias)
    2) bayesian optimization (efficient hyper-parameter tuning)
    3) gradient boosting (flexible and extensive prediction)

    bayesian hyper-parameter optimization is conducted utilizing tree prezen 
    estimation, gradient boosting is conducted utilizing user specified methods
    
    returns custom object depending on the type of prediction
    - regressor: root mean squared error (or other regression metric)
    - classifier: accuracy, prec-recall-f1-support, confusion matrix, roc auc 
    - all cases: feature importance plot, hyperopt trials object
    """
    
    ## conduct input quality checks
    qual_check = ArgumentQualityOptimizer(
        
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
    
    ## return checked arguments (main func args)
    data = qual_check.data
    y = qual_check.y
    loss = qual_check.loss
    k_outer = qual_check.k_outer
    k_inner = qual_check.k_inner
    n_evals = qual_check.n_evals
    seed = qual_check.seed
    verbose = qual_check.verbose
    
    ## return checked arguments (pred func args)
    pred_type = qual_check.pred_type
    method = qual_check.method
    params = qual_check.params
    
    ## suppress warning messages
    wn.filterwarnings(
        action = 'ignore', 
        category = DeprecationWarning
    )
    
    wn.filterwarnings(
        action = 'ignore', 
        category = FutureWarning
    )
    
    ## reset data index
    data.reset_index(
        inplace = True, 
        drop = True
    )
    
    ## test set prediction stores
    y_test_list = []
    y_pred_list = []
    x_test_list = []
    
    if pred_type == "regress":
        rmse_list = []
    
    if pred_type == "multi-class" or pred_type == "binary":
        accu_list = []
        prfs_list = []
        y_labels = np.sort(data[y].unique())
        
        if pred_type == "binary":
            roc_table = pd.DataFrame(columns = ['fold', 'fpr', 'tpr', 'auc'])
    
    ## outer loop k-folds
    k_folds_outer = KFold(
        n_splits = k_outer,
        shuffle = False
    )
    
    ## split data into training-validation and test sets by k-folds
    for train_valid_index, test_index in k_folds_outer.split(data):
        
        ## explanatory features x
        x_train_valid, x_test = data.drop(y, axis = 1).iloc[
            train_valid_index], data.drop(y, axis = 1).iloc[
                test_index]
        
        ## response variable y
        y_train_valid, y_test = data[y].iloc[
            train_valid_index], data[y].iloc[
                test_index]
        
        ## objective function
        def obj_fun(params):
            
            """ objective function to minimize utilizing
            bayesian hyper-parameter optimization """
            
            ## method, params, and objective
            model = method_select(
                pred_type = pred_type,
                method = method,
                params = params,
                loss = loss,
                seed = seed
            )
            
            ## conventional cross-valid for regression
            if pred_type == "regress":
                cv_type = KFold(
                    n_splits = k_inner,
                    random_state = seed,
                    shuffle = False
                )
            
            ## stratified cross-valid for classification
            if pred_type == "multi-class" or pred_type == "binary":
                cv_type = StratifiedKFold(
                    n_splits = k_inner,
                    random_state = seed,
                    shuffle = False
                )
            
            ## inner loop cross-valid
            cv_scores = cross_val_score(
                estimator = model, 
                X = x_train_valid, 
                y = y_train_valid,
                cv = cv_type
            )
            
            ## in rare cases, remove results 
            ## that exceed float range limit
            cv_scores[cv_scores < 1e308]
            cv_scores[cv_scores > -1e308]
            
            ## average the minimized inner loop cross-valid scores
            cv_scores_mean = 1 - np.average(cv_scores)
            
            ## return averaged cross-valid scores and status report
            return {'loss': cv_scores_mean, 'status': STATUS_OK}
        
        ## record results
        trials = Trials()
        
        ## conduct bayesian optimization, inner loop cross-valid
        params_opt = fmin(
            fn = obj_fun,
            space = params,
            algo = tpe.suggest,  ## tree parzen estimation
            max_evals = n_evals,
            trials = trials,
            show_progressbar = verbose
        )
        
        ## modeling method with optimal hyper-params
        model_opt = method_select(
            pred_type = pred_type,
            method = method,
            params = params_opt,
            loss = loss,
            seed = seed
        )
        
        ## train on entire training-validation set
        model_opt = model_opt.fit(
            X = x_train_valid, 
            y = y_train_valid
        )
        
        ## make prediction on test set
        y_pred = model_opt.predict(x_test)
        
        ## store outer cross-validation results
        y_test_list.append(y_test)
        y_pred_list.append(y_pred)
        x_test_list.append(x_test)
        
        ## evaluate global regression performance
        if pred_type == "regress":
            
            ## calculate root mean squared error
            rmse_list.append(
                rmse_score(
                    y_true = y_test,
                    y_pred = y_pred
                )
            )
        
        ## evaulate global classification performance
        if pred_type == "multi-class" or pred_type == "binary":
            
            ## calculate accuracy
            accu_list.append(
                accu_score(
                    y_true = y_test,
                    y_pred = y_pred,
                    normalize = True
                )
            )
            
            ## calculate precision, recall, f1-score, support
            prfs_list.append(
                prfs_score(
                    y_true = y_test,
                    y_pred = y_pred,
                    labels = y_labels,
                    zero_division = 0
                )
            )
            
            ## pre-process roc plot
            if pred_type == "binary":
                fpr, tpr, thres = roc_curve(
                    y_true = y_test,
                    y_score = y_pred
                )
                
                auc = roc_auc_score(
                    y_true = y_test, 
                    y_score = y_pred
                )
                
                roc_table = roc_table.append({
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': auc},
                    
                    ignore_index = True
                )
    
    ## prediction results per prediction type
    if pred_type == "regress":
        return RegressResults(
            trials = trials,
            model = model_opt,
            params = params_opt,
            rmse_list = rmse_list
        )
    
    if pred_type == "multi-class":
        return MultiClassResults(
            trials = trials,
            model = model_opt,
            params = params_opt,
            accu_list = accu_list,
            prfs_list = prfs_list,
            x_data = data.drop(labels = y, axis = 1),
            y_labels = y_labels,
            y_test_list = y_test_list,
            y_pred_list = y_pred_list
        )
    
    if pred_type == "binary":
        return BinaryClassResults(
            trials = trials,
            model = model_opt,
            params = params_opt,
            accu_list = accu_list,
            prfs_list = prfs_list,
            x_data = data.drop(labels = y, axis = 1),
            y_labels = y_labels,
            y_test_list = y_test_list,
            y_pred_list = y_pred_list,
            k_outer = k_outer,
            roc_table = roc_table
        )
