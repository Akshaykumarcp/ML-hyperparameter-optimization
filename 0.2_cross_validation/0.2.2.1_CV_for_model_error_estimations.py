""" CROSS VALIDATION """

# In this example, lets estimate the generalization error of a machine learning model using Cross-Validation Schemes.

import numpy as np
import pandas as pd
from scipy.special import comb
# https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    LeaveOneOut,
    LeavePOut,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

# dataset information: UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    
# in short, classification problem, trying to predict whether the tumor is malignant or benign

# load dataset
breast_cancer_X, breast_cancer_y = load_breast_cancer(return_X_y=True)
X = pd.DataFrame(breast_cancer_X)
y = pd.Series(breast_cancer_y).map({0:1, 1:0})

X.shape
# (569, 30)

X.head()
""" 0      1       2       3        4        5       6        7       8        9       10      11     12  ...       17       18        19     20     21      22      23      24      25      26      27      28       29
0  17.99  10.38  122.80  1001.0  0.11840  0.27760  0.3001  0.14710  0.2419  0.07871  1.0950  0.9053  8.589  ...  0.01587  0.03003  0.006193  25.38  17.33  184.60  2019.0  0.1622  0.6656  0.7119  0.2654  0.4601  0.11890   
1  20.57  17.77  132.90  1326.0  0.08474  0.07864  0.0869  0.07017  0.1812  0.05667  0.5435  0.7339  3.398  ...  0.01340  0.01389  0.003532  24.99  23.41  158.80  1956.0  0.1238  0.1866  0.2416  0.1860  0.2750  0.08902   
2  19.69  21.25  130.00  1203.0  0.10960  0.15990  0.1974  0.12790  0.2069  0.05999  0.7456  0.7869  4.585  ...  0.02058  0.02250  0.004571  23.57  25.53  152.50  1709.0  0.1444  0.4245  0.4504  0.2430  0.3613  0.08758   
3  11.42  20.38   77.58   386.1  0.14250  0.28390  0.2414  0.10520  0.2597  0.09744  0.4956  1.1560  3.445  ...  0.01867  0.05963  0.009208  14.91  26.50   98.87   567.7  0.2098  0.8663  0.6869  0.2575  0.6638  0.17300   
4  20.29  14.34  135.10  1297.0  0.10030  0.13280  0.1980  0.10430  0.1809  0.05883  0.7572  0.7813  5.438  ...  0.01885  0.01756  0.005115  22.54  16.67  152.20  1575.0  0.1374  0.2050  0.4000  0.1625  0.2364  0.07678
 """

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

""" K-Fold Cross-Validation """
# Logistic Regression
logit = LogisticRegression(penalty ='l2', C=10, solver='liblinear', random_state=4, max_iter=10000)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=4)

# estimate generalization error
clf =  cross_validate(
    logit,
    X_train, 
    y_train,
    scoring='accuracy',
    return_train_score=True,
    cv=kf, # k-fold
)

clf['test_score']
# array([0.925     , 0.95      , 0.9625    , 0.96202532, 0.94936709])

clf['train_score']
# array([0.97169811, 0.96540881, 0.96855346, 0.96238245, 0.97178683])

print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
""" 
mean train set accuracy:  0.96859289051872  +-  0.0028086742367865224
mean test set accuracy:  0.9472784810126582  +-  0.014460120917184367 """

""" Repeated K-Fold """

# Logistic Regression
logit = LogisticRegression(penalty ='l2', C=1, solver='liblinear', random_state=4, max_iter=10000)

# Repeated K-Fold Cross-Validation
rkf = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=4,
)

print('We expect K * n performance metrics: ', 5*10)
# We expect K * n performance metrics:  50

# estimate generalization error
clf =  cross_validate(
    logit,
    X_train, 
    y_train,
    scoring='accuracy',
    return_train_score=True,
    cv=rkf, # k-fold
)

print('Number of metrics obtained: ', len(clf['test_score']))
# Number of metrics obtained:  50

clf['test_score']
""" 
array([0.9       , 0.9375    , 0.975     , 0.96202532, 0.94936709,
       0.9625    , 0.9625    , 0.9125    , 0.96202532, 0.92405063,
       0.9875    , 0.95      , 0.975     , 0.91139241, 0.96202532,
       0.95      , 0.9375    , 0.95      , 0.92405063, 0.96202532,
       0.975     , 0.9125    , 0.9375    , 0.94936709, 0.96202532,
       0.9875    , 0.9125    , 0.9375    , 0.91139241, 0.96202532,
       0.9625    , 0.9375    , 0.95      , 0.92405063, 0.93670886,
       0.95      , 0.95      , 0.95      , 0.98734177, 0.88607595,
       0.95      , 0.925     , 0.9625    , 0.96202532, 0.94936709,
       0.925     , 0.9625    , 0.925     , 0.91139241, 0.96202532]) """

print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
""" 
mean train set accuracy:  0.9600488949350366  +-  0.006655639716391218
mean test set accuracy:  0.9457183544303797  +-  0.023400211765523985 """

""" Leave One Out """
# Logistic Regression
logit = LogisticRegression(penalty ='l2', C=1, solver='liblinear', random_state=4, max_iter=10000)

# Leave One Out Cross-Validation
loo = LeaveOneOut()

print('We expect as many metrics as data in the train set: ', len(X_train))
# We expect as many metrics as data in the train set:  398

# estimate generalization error
clf =  cross_validate(
    logit,
    X_train, 
    y_train,
    scoring='accuracy',
    return_train_score=True,
    cv=loo, # k-fold
)

print('Number of metrics obtained: ', len(clf['test_score']))
# Number of metrics obtained:  398

len(clf['test_score'])
# 398

print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
""" 
mean train set accuracy:  0.9575522448514615  +-  0.001238730494304101
mean test set accuracy:  0.9472361809045227  +-  0.22356162123660028 """

""" Leave P Out """
# Logistic Regression
logit = LogisticRegression(penalty ='l2', C=1, solver='liblinear', random_state=4, max_iter=10000)

# Leave P Out Cross-Validation
lpo = LeavePOut(p=2)

# I take a smaller sample of the data, otherwise
# my computer runs out of memory
X_train_small = X_train.head(100)
y_train_small = y_train.head(100)

# The number of combinations of N things taken k at a time.
print('We expect : ', comb(100,2), ' metrics')
# We expect :  4950.0  metrics

# estimate generalization error
clf =  cross_validate(
    logit,
    X_train_small, 
    y_train_small,
    scoring='accuracy',
    return_train_score=True,
    cv=lpo, # k-fold
)

print('Number of metrics obtained: ', len(clf['test_score']))
# Number of metrics obtained:  4950

print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
""" 
mean train set accuracy:  0.9700020614306328  +-  0.0032367717044687024
mean test set accuracy:  0.918989898989899  +-  0.19306928127674033
 """

""" Stratified K-Fold Cross-Validation """
# Logistic Regression
logit = LogisticRegression(penalty ='l2', C=1, solver='liblinear', random_state=4, max_iter=10000)

# Leave P Out Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

# estimate generalization error
clf =  cross_validate(
    logit,
    X_train, 
    y_train,
    scoring='accuracy',
    return_train_score=True,
    cv=skf, # k-fold
)

len(clf['test_score'])
# 5

print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
""" 
mean train set accuracy:  0.9616805662348927  +-  0.004189437253299218
mean test set accuracy:  0.944620253164557  +-  0.02364900328794808 """