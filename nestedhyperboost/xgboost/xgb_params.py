## load libraries
import numpy as np
from hyperopt import hp

## xgboost hyper-params
def xgb_params():

    """ utilized for bayesian hyper-parameter optimization,
    returns xgboost parameter ranges (search space) """

    xgb_params = {

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
            low = 2,
            high = 10,
            q = 1
        ),

        'subsample': hp.uniform(
            label = 'subsample',
            low = 0.50,
            high = 1.00
        ),

        'min_split_loss': hp.uniform(
            label = 'min_split_loss',
            low = 0.00,
            high = 10.00
        ),

        'min_child_weight': hp.quniform(
            label = 'min_child_weight',
            low = 1,
            high = 10,
            q = 1
        ),

        'max_delta_step': hp.quniform(
            label = 'max_delta_step',
            low = 1,
            high = 10,
            q = 1
        ),

        'colsample_bytree': hp.uniform(
            label = 'colsample_bytree',
            low = 0.00,
            high = 1.00
        ),

        'colsample_bylevel': hp.uniform(
            label = 'colsample_bylevel',
            low = 0.00,
            high = 1.00
        ),

        'colsample_bynode': hp.uniform(
            label = 'colsample_bynode',
            low = 0.00,
            high = 1.00
        ),

        'scale_pos_weight': hp.uniform(
            label = 'scale_pos_weight',
            low = 1.00,
            high = 3.00
        ),

        'reg_alpha': hp.uniform(
            label = 'reg_alpha',
            low = 0.00,
            high = 12.00
        ),

        'reg_lambda': hp.uniform(
            label = 'reg_lambda',
            low = 0.00,
            high = 12.00
        )

    }

    return xgb_params
