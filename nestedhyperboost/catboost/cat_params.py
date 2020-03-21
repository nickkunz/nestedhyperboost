## load libraries
import numpy as np
from hyperopt import hp

## catboost hyper-params
def cat_params():
    
    """ utilized for bayesian hyper-parameter optimization, 
    returns catboost parameter ranges (search space) """
    
    cat_params = {
        
        'learning_rate': hp.loguniform(
            label = 'learning_rate', 
            low = np.log(0.001),
            high = np.log(0.300)
        ),
        
        'n_estimators': hp.quniform(
            label = 'n_estimators',
            low = 1,
            high = 1000,
            q = 1
        ),
        
        'max_depth': hp.quniform(
            label = 'max_depth',
            low = 1,
            high = 16,
            q = 1
        ),
        
        'bagging_temperature': hp.uniform(
            label = 'bagging_temperature',
            low = 0.00,
            high = 1.00
        ),
        
        'random_strength': hp.uniform(
            label = 'random_strength',
            low = 0.00,
            high = 10.00
        ),
        
        'fold_permutation_block': hp.quniform(
            label = 'fold_permutation_block',
            low = 1,
            high = 10,
            q = 1
        ),
        
        'fold_len_multiplier': hp.uniform(
            label = 'fold_len_multiplier',
            low = 2.00,
            high = 10.00
        ),
        
        'colsample_bylevel': hp.uniform(
            label = 'colsample_bylevel',
            low = 0.00,
            high = 1.00
        ),
        
        'model_shrink_rate': hp.uniform(
            label = 'model_shrink_rate',
            low = 0.00,
            high = 1.00
        ),
        
        'reg_lambda': hp.uniform(
            label = 'reg_lambda', 
            low = 0.00, 
            high = 12.00
        )
    
    }
    
    return cat_params
