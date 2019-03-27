# Import pandas and scikit preprocessing libraries
from typing import Any, Tuple, Generator, Iterable

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, GridSearchCV, ParameterGrid
from itertools import chain, combinations

# Import models of type Estimator to be used
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Import libraries for time
from datetime import datetime
from datetime import date
import json

def loo_cv(Estimator, parameter_dict, output_filename, sourcefile, length):
    df = pd.read_csv(sourcefile)  # data file
    Xr_t = df.iloc[:, 0:length]
    yr = df['class']
    print(Xr_t.head())  # Check file
    # Scaler for train and test data
    scaler = preprocessing.StandardScaler().fit(Xr_t)
    Xr = scaler.transform(Xr_t)
    # Leave One Out cross validation
    loo = LeaveOneOut()
    param_list = list(ParameterGrid(parameter_dict)) # list of parameters for CV
    filename = output_filename + '.json'
    n_features = Xr.shape[1] # number of columns of array Xr aka num of features
    results = [] # initialize list of results
    subsets = (combinations(range(n_features), k + 1) for k in range(min(n_features, 10)))
    for subsets_k in subsets:  # for each list of subsets of the same size
        for subset in subsets_k:  # for each subset
            Xk = Xr[:, list(subset)] # all rows and selected columns in array
            result_subset = {} # dict init for results in each subset
            accuracy = []
            roc_auc = []
            # Hyperparameter tuning
            for p in param_list:
                df_g = pd.DataFrame(loo.split(Xk, yr))  # Dataframe with train and test index for each split
                df_g = df_g.rename({0: 'Train', 1: 'Test'}, axis='columns')
                y_test = []  # Init test data array
                y_pred = []  # Initialize list for predictions on test data
                y_pred_prob = []  # Initialize list for predictions with probabilities on test data
                for row in df_g.itertuples():
                    train_set_x = Xk[row.Train]
                    train_set_y = yr[row.Train]
                    test_set_x = Xk[row.Test]
                    test_set_y = yr[row.Test]
                    y_test.append(test_set_y.iloc[0])
                    cv_model = Estimator(**p)
                    cv_model.fit(train_set_x, train_set_y)
                    print(cv_model.predict(test_set_x.reshape(1, -1)))
                    print(test_set_y.iloc[0])
                    y_pred.append(cv_model.predict(test_set_x.reshape(1, -1)))
                    try:
                        y_pred_prob.append(cv_model.predict_proba(test_set_x)[:, 1])
                    except AttributeError:
                        y_pred_prob.append(cv_model.decision_function(test_set_x))
                acc = metrics.accuracy_score(y_test, y_pred)
                accuracy.append(acc)
                print("Accuracy", acc)
                roc = metrics.roc_auc_score(y_test, y_pred_prob)
                roc_auc.append(roc)
                # End of parameter tuning for each subset of features
            # Results for each subset
            result_subset['subset'] = subset
            result_subset['roc'] = max(roc_auc)
            result_subset['accuracy'] = max(accuracy)
            result_subset['param']= param_list[roc_auc.index(max(roc_auc))] # note that the chosen params maximize roc
            # result_subset['estimator'] = Estimator
            for key, value in param_list[roc_auc.index(max(roc_auc))].items():
                result_subset[key] = value
            results.append(result_subset)
            # print(result_subset)
        # Save best parameters for each subset in json file for further analysis
    with open(filename, 'w') as fp:
        json.dump(results, fp)

