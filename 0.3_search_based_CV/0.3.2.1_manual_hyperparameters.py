""" 
Manual Search for Hyperparameters
- This is a example to play with the hyperparameters of logistic regression and random forest and get a flavour of 
    how manual search works.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
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

""" Manual Search """
# Logistic Regression
logit = LogisticRegression(penalty ='l2', C=0.001, solver='liblinear', random_state=4, max_iter=10000)

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

# play with C 0.001 and 1
# play with the regularization l1 vs l2

print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
""" 
mean train set accuracy:  0.9170836537134519  +-  0.0038064240947020133
mean test set accuracy:  0.9195886075949368  +-  0.006259426475686005 """

# let's get the predictions
logit.fit(X_train, y_train)

train_preds = logit.predict(X_train)
test_preds = logit.predict(X_test)

print('Train Accuracy: ', accuracy_score(y_train, train_preds))
print('Test Accuracy: ', accuracy_score(y_test, test_preds))
""" 
Train Accuracy:  0.9170854271356784
Test Accuracy:  0.9473684210526315 """


""" Random Forests """

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=0,
    n_jobs=4,
    )

# estimate generalization error
clf =  cross_validate(
    rf,
    X_train, 
    y_train,
    scoring='accuracy',
    return_train_score=True,
    cv=kf, # k-fold
)

# play with n_estimarors 500 and less
# play with max_depth

print('mean train set accuracy: ', np.mean(clf['train_score']), ' +- ', np.std(clf['train_score']))
print('mean test set accuracy: ', np.mean(clf['test_score']), ' +- ', np.std(clf['test_score']))
""" 
mean train set accuracy:  1.0  +-  0.0
mean test set accuracy:  0.9598734177215189  +-  0.020064788507794515 """

# let's get the predictions
rf.fit(X_train, y_train)

train_preds = rf.predict(X_train)
test_preds = rf.predict(X_test)

print('Train Accuracy: ', accuracy_score(y_train, train_preds))
print('Test Accuracy: ', accuracy_score(y_test, test_preds))

""" 
Train Accuracy:  1.0
Test Accuracy:  0.9649122807017544 """