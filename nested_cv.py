# Import pandas and scikit preprocessing libraries
from typing import Any, Tuple, Generator, Iterable

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, GridSearchCV, ParameterGrid
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

#########################################################################################
# Function for nested cross validation with feature selection
#
# Outer CV is 5 fold which reads indexes from pickle file for reproducibility reasons
# Feature selection: results stored for each possible combination
# Inner CV is 10 fold with hyperparameter selection
# All folds are stratified
# Metric used is area under ROC curve from class probabilities or decision function
#
#########################################################################################


def nested_cv(Estimator, parameter_dict, output_filename, sourcefile, length):
    df = pd.read_csv(sourcefile)  # data file
    Xr_t = df.iloc[:, 0:length] # number of features to be used
    yr = df['class'] # class label
    print(Xr_t.head())  # Check file
    outer_folds = pd.read_pickle("DATA/5_fold_split.pkl")  # dataframe with indexes for 5 fold validation
    outer_folds = outer_folds.rename({0: 'Train', 1: 'Test'}, axis='columns')
    # Scaler for train and test data
    scaler = preprocessing.StandardScaler().fit(Xr_t)
    Xr = scaler.transform(Xr_t)
    param_list = list(ParameterGrid(parameter_dict))  # list of parameters for CV
    filename = output_filename + '.json'  # results filename
    n_features = Xr.shape[1]  # number of columns of array Xr = num of features
    results = []  # initialize list of results
    subsets = (combinations(range(n_features), k + 1) for k in range(min(n_features, 10)))
    for subsets_k in subsets:  # for each list of subsets of the same size
        for subset in subsets_k:  # for each subset
            Xk = Xr[:, list(subset)]  # all rows and selected columns in array
            # Loop for 5 fold cross validation = outer folds
            for fold in outer_folds.itertuples(index=True):
                result_subset = {}  # dict init for results in each subset and fold
                result_subset['outer_fold'] = fold.Index
                accuracy = []
                roc_auc = []
                train_set_x_outer = Xk[fold.Train]
                train_set_y_outer = yr[fold.Train]
                test_set_x_outer = Xk[fold.Test]
                test_set_y_outer = yr[fold.Test]
                # Hyperparameter tuning
                for p in param_list:
                    # 10 fold Cross Validation
                    cv = StratifiedKFold(n_splits=10)
                    # Dataframe with train and test index for each split
                    df_g = pd.DataFrame(cv.split(train_set_x_outer, train_set_y_outer))
                    df_g = df_g.rename({0: 'Train', 1: 'Test'}, axis='columns')
                    y_test = []  # Init test data array
                    y_pred = []  # Initialize list for predictions on test data
                    y_pred_prob = []  # Initialize list for predictions with probabilities on test data
                    for row in df_g.itertuples():
                        train_set_x = Xk[row.Train]
                        train_set_y = yr[row.Train]
                        test_set_x = Xk[row.Test]
                        test_set_y = yr[row.Test]
                        for y in test_set_y:
                            y_test.append(y)
                        cv_model = Estimator(**p)
                        cv_model.fit(train_set_x, train_set_y)
                        for yp in cv_model.predict(test_set_x):
                            y_pred.append(yp)
                        try:
                            y_pred_prob_v = cv_model.predict_proba(test_set_x)[:, 1]
                        except AttributeError:
                            y_pred_prob_v = cv_model.decision_function(test_set_x)
                        for ypr in y_pred_prob_v:
                            y_pred_prob.append(ypr)
                    acc = metrics.accuracy_score(y_test, y_pred)
                    accuracy.append(acc)
                    roc = metrics.roc_auc_score(y_test, y_pred_prob)
                    roc_auc.append(roc)
                    # End of parameter tuning for each subset of features and outer fold
                # Results for each subset and k fold
                result_subset['subset'] = subset
                result_subset['roc'] = max(roc_auc)
                result_subset['accuracy'] = accuracy[roc_auc.index(max(roc_auc))]
                result_subset['param'] = param_list[roc_auc.index(max(roc_auc))]  # the chosen params maximize roc
                for key, value in param_list[roc_auc.index(max(roc_auc))].items():
                    result_subset[key] = value
                # Evaluate on test split
                cv_outer = Estimator(**param_list[roc_auc.index(max(roc_auc))])
                cv_outer.fit(train_set_x_outer, train_set_y_outer)
                try:
                    y_pred_prob_outer = cv_outer.predict_proba(test_set_x_outer)[:, 1]
                except AttributeError:
                    y_pred_prob_outer = cv_outer.decision_function(test_set_x_outer)
                roc_outer = metrics.roc_auc_score(test_set_y_outer, y_pred_prob_outer)
                result_subset['outer_roc_test'] = roc_outer
                result_subset['class_predictions'] = list(y_pred_prob_outer)
                # Append to list of results
                results.append(result_subset)
                # Save best parameters for each subset and outer fold in json file for further analysis
    with open(filename, 'w') as fp:
        json.dump(results, fp)
