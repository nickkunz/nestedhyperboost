## load libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from xgboost import plot_importance

## store results
class Results():
    def __init__(self, trials, model, params):
        
        """
        parent to all three results objects: 1) RegressResults,
        2) MultiClassResults, 3) BinaryClassResults
        
        utilized to store bayesian optimized hyper-parameters and model, 
        also creates feature importance plot for all returned objects
        """
        
        self.trials = trials
        self.model = model
        self.params = params
    
    ## feature importance plot
    def feat_plot(self):
        plot_importance(
            booster = self.model,
            max_num_features = 10,
            title = 'Feature Importance',
            color = 'purple',
            grid = True
        )

## store regression results
class RegressResults(Results):
    def __init__(self, trials, model, params, rmse_list):
        super().__init__(trials, model, params)
        
        """
        child to Results object, utilized to calculate and store the average 
        of all outer k-fold cross-validation root mean squared errors, as well
        as all the attributes from the parent Results object, returned to main 
        function ncv_optimizer()
        """
        
        self.rmse_list = rmse_list
    
    ## average rmse results across outer k-folds
    def rmse_mean(self):
        rmse_mean = round(np.average(
            self.rmse_list
            
            ## round results
            ), ndigits = 6
        )
        
        return rmse_mean

## store multi-class classification results
class MultiClassResults(Results):
    def __init__(self, trials, model, params, accu_list, prfs_list,
                 x_data, y_labels, y_test_list, y_pred_list):
        super().__init__(trials, model, params)
        
        """
        child class to Results object, utilized to calculate and store the
        average of all outer k-fold cross-validation accuracy scores, the
        precision-recall-f1-support table, confusion matrix, and confusion 
        matrix plot, as well as all the attributes from the parent Results 
        object
        """
        
        self.accu_list = accu_list
        self.prfs_list = prfs_list
        self.x_data = x_data
        self.y_labels = y_labels
        self.y_test_list = y_test_list
        self.y_pred_list = y_pred_list
    
    ## average accuracy results across outer k-folds
    def accu_mean(self):
        accu_mean = round(np.average(
            self.accu_list
            
            ## round results
            ), ndigits = 6
        )
        
        return accu_mean
    
    ## average prfs results across outer k-folds
    def prfs_mean(self):
        prfs_mean = np.mean(
            a = self.prfs_list,
            axis = 0
        )
        
        ## collate average results
        prfs_table = pd.DataFrame(
            data = prfs_mean.transpose(),
            index = self.y_labels,
            columns = ["precision", "recall", "f1-score", "support"]
        )
        
        return prfs_table
    
    ## create confusion matrix
    def conf_mtrx(self):
        
        ## pre-process confusion matrix
        y_test_list_flat = []
        for i in self.y_test_list:
            for j in i:
                y_test_list_flat.append(j)
        
        ## pre-process confusion matrix
        y_pred_list_flat = []
        for i in self.y_pred_list:
            for j in i:
                y_pred_list_flat.append(j)
        
        ## create confusion matrix
        conf_mtrx = confusion_matrix(
            y_true = y_test_list_flat,
            y_pred = y_pred_list_flat,
            labels = self.y_labels,
            normalize = None
        )
        
        ## collate average results
        conf_mtrx_table = pd.DataFrame(
            data = conf_mtrx,
            index = self.y_labels,
            columns = self.y_labels
        )
        
        return conf_mtrx_table
    
    ## create confusion matrix plot
    def conf_mtrx_plot(self):
        
        ## pre-process confusion matrix
        y_test_list_flat = []
        for i in self.y_test_list:
            for j in i:
                y_test_list_flat.append(j)
        
        ## plot confusion matrix
        plot_confusion_matrix(
            estimator = self.model,
            X = self.x_data,
            y_true = y_test_list_flat,
            labels = self.y_labels,
            cmap = 'plasma',
            normalize = None,
            
            ## disable scientific notation
            values_format = '.0f'
        )
        
        plt.title('Confusion Matrix')

## store binary classification results
class BinaryClassResults(MultiClassResults):
    def __init__(self, trials, model, params, accu_list, prfs_list, x_data, 
                 y_labels, y_test_list, y_pred_list, k_outer, roc_table):
        super().__init__(trials, model, params, accu_list, prfs_list, 
                         x_data, y_labels, y_test_list, y_pred_list)
        
        """
        child class to MultiClassResults object, utilized to calculate and 
        store the roc curve plot, as well as all the attributes from the 
        parent MultiClassResults object, returned to main function 
        ncv_optimizer() when 'pred' argument is specified "binary"
        """
        
        self.k_outer = k_outer
        self.roc_table = roc_table
    
    ## create roc curve plot
    def roc_curve_plot(self):
        
        ## pre-process roc curve
        self.roc_table['fold'] = range(1, self.k_outer + 1)
        self.roc_table.set_index('fold', inplace = True)
        
        ## plot settings
        plt.figure(figsize = (6, 6))
        plt.margins(y = 0, x = 0)
        plt.grid()
        
        ## plot roc curve 
        for i in self.roc_table.index:
            plt.plot(
                self.roc_table.loc[i]['fpr'], 
                self.roc_table.loc[i]['tpr'], 
                label = "Fold {}, AUC = {:.3f}".format(
                    i, self.roc_table.loc[i]['auc']
                )
            )
        
        ## baseline
        plt.plot(
            [0,1], [0,1], 
            color = 'black', 
            linestyle = '--',
            linewidth = 1
        )
        
        ## title and legend
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc = 'lower right')
        
        ## y-axis labels
        plt.yticks(np.arange(0.0, 1.1, step = 0.1))
        plt.ylabel("True Positive Rate")
        
        ## x-axis labels
        plt.xticks(np.arange(0.0, 1.1, step = 0.1))
        plt.xlabel("False Positive Rate")
