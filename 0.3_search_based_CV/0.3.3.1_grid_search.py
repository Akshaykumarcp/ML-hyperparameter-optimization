""" Grid Search for Hyperparameters """

import pandas as pd
import matplotlib.pyplot as plt
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)

# in short, classification problem, trying to predict whether the tumor is malignant or benign

# load dataset
breast_cancer_X, breast_cancer_y = load_breast_cancer(return_X_y=True)

X = pd.DataFrame(breast_cancer_X)
y = pd.Series(breast_cancer_y).map({0:1, 1:0})

X.head()
""" 
      0      1       2       3        4        5       6        7       8        9       10      11  ...       18        19     20     21      22      23      24      25      26      27      28       29
0  17.99  10.38  122.80  1001.0  0.11840  0.27760  0.3001  0.14710  0.2419  0.07871  1.0950  0.9053  ...  0.03003  0.006193  25.38  17.33  184.60  2019.0  0.1622  0.6656  0.7119  0.2654  0.4601  0.11890   
1  20.57  17.77  132.90  1326.0  0.08474  0.07864  0.0869  0.07017  0.1812  0.05667  0.5435  0.7339  ...  0.01389  0.003532  24.99  23.41  158.80  1956.0  0.1238  0.1866  0.2416  0.1860  0.2750  0.08902   
2  19.69  21.25  130.00  1203.0  0.10960  0.15990  0.1974  0.12790  0.2069  0.05999  0.7456  0.7869  ...  0.02250  0.004571  23.57  25.53  152.50  1709.0  0.1444  0.4245  0.4504  0.2430  0.3613  0.08758   
3  11.42  20.38   77.58   386.1  0.14250  0.28390  0.2414  0.10520  0.2597  0.09744  0.4956  1.1560  ...  0.05963  0.009208  14.91  26.50   98.87   567.7  0.2098  0.8663  0.6869  0.2575  0.6638  0.17300   
4  20.29  14.34  135.10  1297.0  0.10030  0.13280  0.1980  0.10430  0.1809  0.05883  0.7572  0.7813  ...  0.01756  0.005115  22.54  16.67  152.20  1575.0  0.1374  0.2050  0.4000  0.1625  0.2364  0.07678   

[5 rows x 30 columns] """

# percentage of benign (0) and malign tumors (1)

y.value_counts() / len(y)
""" 
0    0.627417
1    0.372583
dtype: float64 """
# split dataset into a train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape
# ((398, 30), (171, 30))

""" Grid Search """

# Let's use Grid Search to find the best hyperparameters for a Gradient Boosting Classifier.

# set up the model
gbm = GradientBoostingClassifier(random_state=0)

# determine the hyperparameter space
param_grid = dict(
    n_estimators=[10, 20, 50, 100],
    min_samples_split=[0.1, 0.3, 0.5],
    max_depth=[1,2,3,4,None],
    )

print('Number of hyperparam combinations: ', len(param_grid['n_estimators'])*len(param_grid['min_samples_split'])*len(param_grid['max_depth']))
# Number of hyperparam combinations:  60

# set up the search
search = GridSearchCV(gbm, param_grid, scoring='roc_auc', cv=5, refit=True)

# find best hyperparameters
search.fit(X_train, y_train)
GridSearchCV(cv=5, estimator=GradientBoostingClassifier(random_state=0),
             param_grid={'max_depth': [1, 2, 3, 4, None],
                         'min_samples_split': [0.1, 0.3, 0.5],
                         'n_estimators': [10, 20, 50, 100]},
             scoring='roc_auc')

# the best hyperparameters are stored in an attribute
search.best_params_
# {'max_depth': 2, 'min_samples_split': 0.1, 'n_estimators': 100}

# we also find the data for all models evaluated
results = pd.DataFrame(search.cv_results_)

print(results.shape)
# (60, 16)

results.head()
""" 
   mean_fit_time  std_fit_time  mean_score_time  std_score_time param_max_depth  ... split3_test_score split4_test_score mean_test_score  std_test_score  rank_test_score
0       0.010000  2.780415e-07         0.001600        0.000800               1  ...          0.983103          0.940136        0.964248        0.016026               58
1       0.018000  1.784161e-07         0.001400        0.000490               1  ...          0.986897          0.970068        0.971193        0.011607               52
2       0.042800  1.599980e-03         0.002000        0.000001               1  ...          0.993103          0.980272        0.983475        0.011647               33
3       0.081746  3.862444e-04         0.002007        0.000013               1  ...          0.997241          0.983673        0.988783        0.009298               18
4       0.010002  1.974129e-06         0.001792        0.000411               1  ...          0.983103          0.940136        0.964248        0.016026               58

[5 rows x 16 columns] """

# we can order the different models based on their performance
results.sort_values(by='mean_test_score', ascending=False, inplace=True)
results.reset_index(drop=True, inplace=True)
results[[
    'param_max_depth', 'param_min_samples_split', 'param_n_estimators',
    'mean_test_score', 'std_test_score',
]].head()
""" 
  param_max_depth param_min_samples_split param_n_estimators  mean_test_score  std_test_score
0               2                     0.1                100         0.992415        0.006426
1               2                     0.3                100         0.992013        0.006461
2               3                     0.5                100         0.991949        0.006547
3               4                     0.5                100         0.991620        0.007117
4               2                     0.5                100         0.991545        0.006363 """

results[[
    'param_max_depth', 'param_min_samples_split', 'param_n_estimators',
    'mean_test_score', 'std_test_score',
]].tail()
""" 
   param_max_depth param_min_samples_split param_n_estimators  mean_test_score  std_test_score
55               2                     0.3                 10         0.971111        0.011933
56               2                     0.5                 10         0.965857        0.009779
57               1                     0.3                 10         0.964248        0.016026
58               1                     0.5                 10         0.964248        0.016026
59               1                     0.1                 10         0.964248        0.016026 """

# plot model performance and error
results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)
plt.ylabel('Mean test score')
plt.xlabel('Hyperparameter combinations')
plt.show()

X_train_preds = search.predict_proba(X_train)[:,1]
X_test_preds = search.predict_proba(X_test)[:,1]

print('Train roc_auc: ', roc_auc_score(y_train, X_train_preds))
print('Test roc_auc: ', roc_auc_score(y_test, X_test_preds))
""" 
Train roc_auc:  1.0
Test roc_auc:  0.996766607877719 """

# let's make a function to evaluate the model performance based on
# single hyperparameters

def summarize_by_param(hparam):
    
    tmp = pd.concat([
        results.groupby(hparam)['mean_test_score'].mean(),
        results.groupby(hparam)['mean_test_score'].std(),
    ], axis=1)

    tmp.columns = ['mean_test_score', 'std_test_score']
    
    return tmp

# performance change for n_estimators
tmp = summarize_by_param('param_n_estimators')

tmp.head()
""" 
                    mean_test_score  std_test_score
param_n_estimators
10                         0.973527        0.006351
20                         0.980913        0.005263
50                         0.987444        0.002510
100                        0.990123        0.002077 """

tmp['mean_test_score'].plot(yerr=[tmp['std_test_score'], tmp['std_test_score']], subplots=True)
plt.ylabel('roc-auc')
plt.show()

# The optimal hyperparameter seems to be somewhere between 60 and 100.

tmp = summarize_by_param('param_max_depth')
tmp['mean_test_score'].plot(yerr=[tmp['std_test_score'], tmp['std_test_score']], subplots=True)
plt.ylabel('roc-auc')
plt.show()

# The optimal hyperparameter seems to be between 2 and 3.

tmp = summarize_by_param('param_min_samples_split')
tmp['mean_test_score'].plot(yerr=[tmp['std_test_score'], tmp['std_test_score']], subplots=True)
# array([<AxesSubplot:xlabel='param_min_samples_split'>], dtype=object)
plt.show()

# This parameter does not seem to improve performance much.

# determine the hyperparameter space
param_grid = dict(
    n_estimators=[60, 80, 100, 120],
    max_depth=[2,3],
    loss = ['deviance', 'exponential'],
    )

# set up the search
search = GridSearchCV(gbm, param_grid, scoring='roc_auc', cv=5, refit=True)

# find best hyperparameters
search.fit(X_train, y_train)
GridSearchCV(cv=5, estimator=GradientBoostingClassifier(random_state=0),
             param_grid={'loss': ['deviance', 'exponential'],
                         'max_depth': [2, 3],
                         'n_estimators': [60, 80, 100, 120]},
             scoring='roc_auc')
# the best hyperparameters are stored in an attribute

search.best_params_
# {'loss': 'exponential', 'max_depth': 2, 'n_estimators': 120}
results = pd.DataFrame(search.cv_results_)
results.sort_values(by='mean_test_score', ascending=False, inplace=True)
results.reset_index(drop=True, inplace=True)
results[[
    'param_max_depth', 'param_loss', 'param_n_estimators',
    'mean_test_score', 'std_test_score',
]].head(8)
""" 
  param_max_depth   param_loss param_n_estimators  mean_test_score  std_test_score
0               2  exponential                120         0.993095        0.006174
1               2  exponential                100         0.992828        0.006021
2               2  exponential                 80         0.992765        0.006340
3               2     deviance                120         0.992556        0.006791
4               2     deviance                100         0.992149        0.006904
5               2  exponential                 60         0.991210        0.007305
6               2     deviance                 60         0.991066        0.006638
7               2     deviance                 80         0.990797        0.006979 """
 
results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)

plt.ylabel('Mean test score')
plt.xlabel('Hyperparameter combinations')
plt.show()

X_train_preds = search.predict_proba(X_train)[:,1]
X_test_preds = search.predict_proba(X_test)[:,1]

print('Train roc_auc: ', roc_auc_score(y_train, X_train_preds))
print('Test roc_auc: ', roc_auc_score(y_test, X_test_preds))
""" 
Train roc_auc:  0.9999999999999999
Test roc_auc:  0.9973544973544973 """