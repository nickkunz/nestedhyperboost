## load libraries
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

## boosting method (xgboost, lightgbm, catboost)
def method_select(pred_type, method, params, loss, seed):
    
    ## xgboost
    if method in [XGBRegressor, XGBClassifier]:
        
        ## hyper-param specification
        method_params = method(
            
            ## learned params
            learning_rate = params["learning_rate"],
            n_estimators = int(params["n_estimators"]),
            max_depth = int(params["max_depth"]),
            subsample = params["subsample"],
            min_split_loss = params["min_split_loss"],
            min_child_weight = int(params["min_child_weight"]),
            max_delta_step = int(params["max_delta_step"]),
            colsample_bytree = params["colsample_bytree"],
            colsample_bylevel = params["colsample_bylevel"],
            colsample_bynode = params["colsample_bynode"],
            scale_pos_weight = params["scale_pos_weight"],
            reg_alpha = params["reg_alpha"],
            reg_lambda = params["reg_lambda"],
            
            ## specified params
            booster = "gbtree",  ## gradient tree boosting
            objective = loss,
            random_state = seed,
            silent = True
        )
    
    ## lightgbm
    if method in [LGBMRegressor, LGBMClassifier]:
        
        ## hyper-param specification
        method_params = method(
            
            ## learned params
            num_leaves = int(params["num_leaves"]),
            learning_rate = params["learning_rate"],
            n_estimators = int(params["n_estimators"]),
            subsample = params["subsample"],
            subsample_for_bin = int(params["subsample_for_bin"]),
            subsample_freq = int(params["subsample_freq"]),
            min_split_gain = params["min_split_gain"],
            min_child_samples = int(params["min_child_samples"]),
            colsample_bytree = params["colsample_bytree"],
            reg_alpha = params["reg_alpha"],
            reg_lambda = params["reg_lambda"],
            
            ## specified params
            boosting_type = "goss",  ## gradient based one-sided sampling
            objective = loss,
            random_state = seed,
            silent = True
        )
    
    ## catboost
    if method in [CatBoostRegressor, CatBoostClassifier]:
        
        ## hyper-param specification
        method_params = method(
            
            ## learned params
            learning_rate = params["learning_rate"],
            n_estimators = int(params["n_estimators"]),
            max_depth = int(params["max_depth"]),
            bagging_temperature = params["bagging_temperature"],
            random_strength = params["random_strength"],
            fold_permutation_block = int(params["fold_permutation_block"]),
            fold_len_multiplier = params["fold_len_multiplier"],
            colsample_bylevel = params["colsample_bylevel"],
            model_shrink_rate = params["model_shrink_rate"],
            reg_lambda = params["reg_lambda"],
            
            ## specified params
            boosting_type = "Ordered",  ## ordered boosting
            objective = loss,
            random_state = seed,
            verbose = False
        )
    
    ## returns boosting method and params
    return method_params
