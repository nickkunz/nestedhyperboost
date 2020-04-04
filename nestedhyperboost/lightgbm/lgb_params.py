## load libraries
import numpy as np
from hyperopt import hp

## lightgbm hyper-params
def lgb_params():

    """ utilized for bayesian hyper-parameter optimization,
    returns lightgbm parameter ranges (search space) """

    lgb_params = {

        'num_leaves': hp.quniform(
            label = 'num_leaves',
            low = 2,
            high = 100,
            q = 1
        ),

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

        'subsample': hp.uniform(
            label = 'subsample',
            low = 0.00,
            high = 1.00
        ),

        'subsample_for_bin': hp.quniform(
            label = 'subsample_for_bin',
            low = 1,
            high = 500000,
            q = 1
        ),

        'subsample_freq': hp.uniform(
            label = 'subsample_freq',
            low = 0.00,
            high = 1.00
        ),

        'min_split_gain': hp.uniform(
            label = 'min_split_gain',
            low = 0.00,
            high = 1.00
        ),

        'min_child_samples': hp.quniform(
            label = 'min_child_samples',
            low = 1,
            high = 100,
            q = 1
        ),

        'colsample_bytree': hp.uniform(
            label = 'colsample_bytree',
            low = 0.00,
            high = 1.00
        ),

        'reg_lambda': hp.uniform(
            label = 'reg_lambda',
            low = 0.00,
            high = 2.00
        ),

        'reg_alpha': hp.uniform(
            label = 'reg_alpha',
            low = 0.00,
            high = 2.00
        )

    }

    return lgb_params
