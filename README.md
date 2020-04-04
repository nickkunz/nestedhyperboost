<div align="center">
  <img src="https://github.com/nickkunz/nestedhyperboost/blob/master/media/images/nestedhyperboost_banner.png">
</div>

## Nested Cross-Validation for Bayesian Optimized Gradient Boosting
[![PyPI version](https://badge.fury.io/py/nestedhyperboost.svg)](https://badge.fury.io/py/nestedhyperboost)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.com/nickkunz/nestedhyperboost.svg?branch=master)](https://travis-ci.com/nickkunz/nestedhyperboost)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8d3b4a3d156c4c7f9c62ac540782efd6)](https://app.codacy.com/manual/nickkunz/nestedhyperboost?utm_source=github.com&utm_medium=referral&utm_content=nickkunz/nestedhyperboost&utm_campaign=Badge_Grade_Dashboard)
![GitHub last commit](https://img.shields.io/github/last-commit/nickkunz/nestedhyperboost)

## Description
A Python implementation that unifies Nested K-Fold Cross-Validation, Bayesian Hyperparameter Optimization, and Gradient Boosting. Designed for rapid prototyping on small to mid-sized data sets (can be manipulated within memory). Quickly obtains high quality prediction results by abstracting away tedious hyperparameter tuning and implementation details in favor of usability and implementation speed. Bayesian Hyperparamter Optimization utilizes Tree Parzen Estimation (TPE) from the <a href="https://github.com/hyperopt/hyperopt">Hyperopt</a> package. Gradient Boosting can be conducted one of three ways. Select between <a href="https://github.com/dmlc/xgboost">XGBoost</a>, <a href="https://github.com/microsoft/LightGBM">LightGBM</a>, or <a href="https://github.com/catboost/catboost">CatBoost</a>. <a href="https://github.com/dmlc/xgboost">XGBoost</a> is applied using traditional Gradient Tree Boosting (GTB). <a href="https://github.com/microsoft/LightGBM">LightGBM</a> is applied using its novel Gradient Based One Sided Sampling (GOSS). <a href="https://github.com/catboost/catboost">CatBoost</a> is applied usings its novel Ordered Boosting. NestedHyperBoost can be applied to regression, multi-class classification, and binary classification problems.

## Features
1. Consistent syntax across all Gradient Boosting methods.
2. Supported Gradient Boosting methods: <a href="https://github.com/dmlc/xgboost">XGBoost</a>, <a href="https://github.com/microsoft/LightGBM">LightGBM</a>, <a href="https://github.com/catboost/catboost">CatBoost</a>.
3. Returns custom object that includes common performance metrics and plots.
4. Developed for readability, maintainability, and future improvement.

## Requirements
1. Python 3
2. NumPy
3. Pandas
4. MatPlotLib
5. Scikit-Learn
6. Hyperopt
7. XGBoost
8. LightGBM
9. CatBoost

## Installation
```python
## install pypi release
pip install nestedhyperboost

## install developer version
pip install git+https://github.com/nickkunz/nestedhyperboost.git
```

## Usage
```python
## load libraries
from nestedhyperboost import xgboost
from sklearn import datasets
import pandas

## load data
data_sklearn = datasets.load_iris()
data = pandas.DataFrame(data_sklearn.data, columns = data_sklearn.feature_names)
data['target'] = pandas.Series(data_sklearn.target)

## conduct nestedhyperboost
results = xgboost.xgb_ncv_classifier(
    data = data,
    y = 'target',
    k_inner = 5,
    k_outer = 5,
    n_evals = 10
)

## preview results
results.accu_mean()
results.conf_mtrx()
results.prfs_mean()

## preview plots
results.feat_plot()

## model and params
model = results.model
params = results.params
```

## License
© Nick Kunz, 2019. Licensed under the General Public License v3.0 (GPLv3).

## Contributions
NestedHyperBoost is open for improvements and maintenance. Your help is valued to make the package better for everyone.

## References
Bergstra, J., Bardenet, R., Bengio, Y., Kegl, B. (2011). Algorithms for Hyper-Parameter Optimization. https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf.

Bergstra, J., Yamins, D., Cox, D. D. (2013). Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. 
Proceedings of the 30th International Conference on International Conference on Machine Learning. 28:I115–I123. 
http://proceedings.mlr.press/v28/bergstra13.pdf.

Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 785–794.
https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf.

Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Proceedings of the 31st International Conference on Neural Information Processing Systems. 3146-3154. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf.

Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. Proceedings of the 32nd International Conference on Neural Information Processing Systems. 6639–6649.
http://learningsys.org/nips17/assets/papers/paper_11.pdf.
