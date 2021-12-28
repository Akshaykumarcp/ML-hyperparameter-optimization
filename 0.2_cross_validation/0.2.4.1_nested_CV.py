""" 
Nested Cross-Validation
- In this example, we will implement nested cross-validation to both select the best hyperparameters and obtain a 
    better estimate of the generalization error of the final model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer # https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import (
    KFold,
    GridSearchCV,
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape
# ((398, 30), (171, 30))

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

""" Nested Cross-Validation """
def nested_cross_val(model, grid):

    # configure the outer loop cross-validation procedure
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)

    # configure the inner loop cross-validation procedure
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)

    # enumerate splits
    outer_results = list()
    inner_results = list()

    for train_ix, test_ix in cv_outer.split(X_train):

        # split data
        xtrain, xtest = X_train.loc[train_ix, :], X_train.loc[test_ix, :]
        ytrain, ytest = y_train[train_ix], y_train[test_ix]

        # define search
        search = GridSearchCV(
            model, grid, scoring='accuracy', cv=cv_inner, refit=True)

        # execute search
        search.fit(xtrain, ytrain)

        # evaluate model on the hold out dataset
        yhat = search.predict(xtest)

        # evaluate the model
        accuracy = accuracy_score(ytest, yhat)

        # store the result
        outer_results.append(accuracy)
        
        inner_results.append(search.best_score_)

        # report progress
        print(' >> accuracy_outer=%.3f, accuracy_inner=%.3f, cfg=%s' %
              (accuracy, search.best_score_, search.best_params_))

    # summarize the estimated performance of the model
    print()
    print('accuracy_outer: %.3f +- %.3f' %
          (np.mean(outer_results), np.std(outer_results)))
    print('accuracy_inner: %.3f +- %.3f' %
          (np.mean(inner_results), np.std(inner_results)))

    return search.fit(X_train, y_train)

""" Logistic Regression """
logit = LogisticRegression(
    penalty ='l2', C=1, solver='liblinear', random_state=4, max_iter=10000)

# hyperparameter space
logit_param = dict(
    penalty=['l1', 'l2'],
    C=[0.1, 1, 10],
)
logit_search = nested_cross_val(logit, logit_param)
""" 
 >> accuracy_outer=0.975, accuracy_inner=0.950, cfg={'C': 10, 'penalty': 'l1'}
 >> accuracy_outer=0.963, accuracy_inner=0.947, cfg={'C': 10, 'penalty': 'l1'}
 >> accuracy_outer=0.975, accuracy_inner=0.959, cfg={'C': 10, 'penalty': 'l1'}
 >> accuracy_outer=0.962, accuracy_inner=0.959, cfg={'C': 10, 'penalty': 'l1'}
 >> accuracy_outer=0.937, accuracy_inner=0.959, cfg={'C': 10, 'penalty': 'l1'}

accuracy_outer: 0.962 +- 0.014
accuracy_inner: 0.955 +- 0.005 

The generalization error of the values obtained in the inner loop, is smaller, thus it is optimistically biased.

"""


# let's get the predictions

X_train_preds = logit_search.predict(X_train)
X_test_preds = logit_search.predict(X_test)

# let's examine the accuracy
print('Train accuracy: ', accuracy_score(y_train, X_train_preds))
print('Test accuracy: ', accuracy_score(y_test, X_test_preds))

""" 
Train accuracy:  0.9748743718592965
Test accuracy:  0.9707602339181286
Note how the accuracy of the model in the train set falls within the interval estimated in the outer loop, but outside the interval estimated with the inner loop. """

""" Random Forests """
rf_param = dict(
    n_estimators=[10, 50, 100, 200],
    min_samples_split=[0.1, 0.3, 0.5, 1.0],
    max_depth=[1,2,3,None],
    )

rf = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=2,
    max_depth=3,
    random_state=0,
    n_jobs=-1,
    )
rf_search = nested_cross_val(rf, rf_param)
""" 
 >> accuracy_outer=0.963, accuracy_inner=0.946, cfg={'max_depth': None, 'min_samples_split': 0.1, 'n_estimators': 200}
 >> accuracy_outer=0.950, accuracy_inner=0.953, cfg={'max_depth': 2, 'min_samples_split': 0.1, 'n_estimators': 10}
 >> accuracy_outer=0.938, accuracy_inner=0.956, cfg={'max_depth': None, 'min_samples_split': 0.1, 'n_estimators': 100}
 >> accuracy_outer=0.987, accuracy_inner=0.934, cfg={'max_depth': 2, 'min_samples_split': 0.1, 'n_estimators': 50}
 >> accuracy_outer=0.911, accuracy_inner=0.959, cfg={'max_depth': 2, 'min_samples_split': 0.1, 'n_estimators': 200}

accuracy_outer: 0.950 +- 0.025
accuracy_inner: 0.950 +- 0.009

The generalization error of the values obtained in the inner loop, is smaller, thus it is optimistically biased. """

# let's get the predictions
X_train_preds = rf_search.predict(X_train)
X_test_preds = rf_search.predict(X_test)

# let's examine the accuracy
print('Train accuracy: ', accuracy_score(y_train, X_train_preds))
print('Test accuracy: ', accuracy_score(y_test, X_test_preds))
""" 
Train accuracy:  0.9824120603015075
Test accuracy:  0.9473684210526315
"""