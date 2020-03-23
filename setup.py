from setuptools import setup

setup(
    name = 'nestedhyperboost',
    version = '0.0.1',
    description = 'A wrapper for conducting Nested Cross-Validation with Bayesian Hyper-Parameter Optimized Gradient Boosting',
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    author = 'Nick Kunz',
    author_email = 'nick.kunz@me.com',
    url = 'https://github.com/nickkunz/nestedhyperboost',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        ],
    keywords = [
        'nested cross-validation',
        'bayesian optimization',
        'gradient boosting',
        'xgboost',
        'lightgbm',
        'catboost'
    ],
    packages = ['nestedhyperboost'],
    include_package_data = False,
    install_requires = [
        'numpy',
        'pandas',
        'matplotlib',
        'sklearn',
        'hyperopt',
        'xgboost',
        'lightgbm',
        'catboost'
    ],
    tests_require = ['nose'],
    test_suite = 'nose.collector'
)
