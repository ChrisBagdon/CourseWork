#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 2
    See <https://snlp2020.github.io/a2/> for detailed instructions.

    Statistical Language Processing (SNLP), Assignment 1
    See <https://snlp2020.github.io/a1/> for detailed instructions.

 Course:      SNLP Summer 2020
 Assignment:  (lab 2)
 Author:      (Christopher Bagdon)
 Description: (Script to read in dataset, calculate correlation, fit and tune regression model)
 Honor Code:  I pledge that this program represents my own work.

"""
import csv
import gzip
import statistics

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



def read_data(filename, shuffle=True):
    """ Read the tab-separated training or test files.

    Parameters:
    -----------
    filename:  The name of the file to read.
    shuffle:   If true, the instances should be shuffled (make sure
               not scramble correspondences between the predictors and
               the outcomes).

    Returns: (a tuple)
    -----------
    age:        A 1D numpy array with the first (age) column.
    features:   A 2D (n_instances x n_features) with numeric features
                (including the document length, which should be the first
                column.)
    """

    with gzip.open(filename) as file:
        data = np.genfromtxt(file, delimiter="\t", skip_header=1)
        # Shuffle if required by arg
        if shuffle:
            np.random.shuffle(data)
        # Split prediction values from features
        rawage = data[:, 0]
        rawfeat = data[:, 1:]

    return (rawage, rawfeat)


def correlate(xseq, yseq):
    """Return the correlation coefficient between the given sequences.
    """
    leng = len(xseq)
    # Calculate mean
    ux = sum(xseq) / leng
    uy = sum(yseq) / leng
    # Calculate variance
    xvar = sum((x - ux) ** 2 for x in xseq) / leng
    yvar = sum((y - uy) ** 2 for y in yseq) / leng
    # Calculate covariance
    covar = sum([(xseq[i] - ux) * (yseq[i] - uy)
                 for i in range(len(xseq))]) / leng
    # Calculate correlation
    corre = covar / ((xvar ** 0.5) * (yvar ** 0.5))

    return corre


def regression_scores(x_train, y_train, x_dev=None, y_dev=None, lambda_=None):
    """Train a regression model and return rmse and r2.
    
    If x_dev and y_dev are None, scores are calculated on the
    training data, otherwise on the development (validation) data.

    Parameters:
    -----------
    x_train:   Feature matrix for training (A 2D numpy array)
    y_train:   The values to predict.
    x_dev:     Same as x_train, for development data.
    y_dev:     Same as y_train, for development data.
    lambda_:   If None, train a least-squares regression model,
               otherwise train a L2-regularized regression (Ridge)
               with given regularization strength.

    Returns: (a tuple)
    -----------
    rmse:       Root mean squared error.
    r-squared:  The coefficient of determination (R-squared)
    """
    ### The main code for 2.3 and 2.4 goes here

    # Choose to use L2 reg or not
    if lambda_ is None:
        regr = linear_model.LinearRegression()
    else:
        regr = linear_model.Ridge(lambda_)

    regr.fit(x_train, y_train)
    # Choose to use dev set or not
    if x_dev is None or y_dev is None:
        # Generate predictions
        y_pred = regr.predict(x_train)
        # Calculate RMSE and r2 score
        rmse = mean_squared_error(y_train, y_pred) ** 0.5
        r2 = r2_score(y_train, y_pred)
    else:
        y_pred = regr.predict(x_dev)
        rmse = mean_squared_error(y_dev, y_pred) ** 0.5
        r2 = r2_score(y_dev, y_pred)

    return (rmse, r2)


def tune(x_train, y_train, x_dev, y_dev):
    """Tune the regularization strength of a L2-regularized regression model.

    You are free to choose the method the range to search. Most
    straightforward method is to use a "grid search", calculating the
    r2 values in a range with a small step, and returning the constant
    that yields the best r2 score. However, there are more interesting
    and/or more efficient ways to look for the best hyperparameter.

    Parameters:
    -----------
    x_train:   Feature matrix for training (A 2D numpy array)
    y_train:   The values to predict.
    x_dev:     Same as x_train, for development data.
    y_dev:     Same as y_train, for development data.

    """
    ### Exercise 2.5
    best_r2 = 0
    best_val = 0

    # In order to do 1000s of reshuffles I load in the raw data once here
    # Rather than using read_data everytime
    with gzip.open(train_file) as file:
        data = np.genfromtxt(file, delimiter="\t", skip_header=1)

    # Range here can be easily adjusted to tune with greater detail
    # If I were to write this method's sig, I would make the range and arg
    # At first I covered a much broader range, but the current value is what
    # I tuned down to
    for val in range(0,130,1):
        # Range only steps in ints, so this multiplier allows float values
        val = val*0.005
        result_list = []
        # Does reshuffles per value. Each shuffle is compared to the r2 w/o L2
        # The average difference in r2 score is recorded
        for x in range(0,100):
            np.random.shuffle(data)
            age = data[:, 0]
            feats = data[:, 1:]
            train_age, dev_age = np.split(age, [int(0.8 * len(age))])
            train_feat, dev_feat = np.split(feats, [int(0.8 * len(feats))])
            no_reg_r2 = regression_scores(train_feat, train_age, dev_feat, dev_age)[1]
            result = regression_scores(train_feat, train_age, dev_feat, dev_age, lambda_=val)
            result_list.append(result[1]-no_reg_r2)
            av_r2_dif = statistics.mean(result_list)
        # If the average of 100 shuffles for the current value is better than the previous, replace
        if av_r2_dif > best_r2:
            best_r2 = av_r2_dif
            best_val = val
            # I found it very helpful to be able to see the progress by watching this printout
            print("Biggest r2 difference is {} from val {}".format(best_r2, best_val))
    return best_val

if __name__ == "__main__":
    train_file = 'train.tsv.gz'
    test_file = 'test.tsv.gz'

    ### E2.1 - read the data
    age, feats = read_data(train_file)
    ### train/dev splitting goes below the code below assumes that
    ### they are prefixed with train_ and dev_ respectively
    train_age, dev_age = np.split(age, [int(0.8 * len(age))])
    train_feat, dev_feat = np.split(feats, [int(0.8 * len(feats))])
    ### E2.2
    print("Correlation between age and length: {}.".format(
        correlate(age, feats[:, 0])))
    print("From numpy                          {}.".format(
        np.corrcoef(age, feats[:, 0])[0, 1]))

    ### E2.3 - calculate (using print regression_scores()) and print
    ###        out the required scores below
    print("Full set, w/o dev set: RMSE: {0[0]} r2: {0[1]}".format(
            regression_scores(train_feat, train_age)))
    print("Full set, w/ dev set: RMSE: {0[0]} r2: {0[1]}".format(
            regression_scores(train_feat, train_age, dev_feat, dev_age)))

    # Split to only have length feature
    length_feat_train = train_feat[:, 0].reshape(-1, 1)
    length_feat_dev = dev_feat[:, 0].reshape(-1, 1)

    # Try again with only length feature
    print("Length only, w/o dev set: RMSE: {0[0]} r2: {0[1]}".format(
        regression_scores(length_feat_train, train_age)))
    print("Length only, w/ dev set: RMSE: {0[0]} r2: {0[1]}".format(
        regression_scores(length_feat_train, train_age, length_feat_dev, dev_age)))

    ### E2.4 - calculate (using print regression_scores()) and print
    ###        out the required scores below
    print("L2 Reg lambda 1, w/o dev set: RMSE: {0[0]} r2: {0[1]}".format(
        regression_scores(train_feat, train_age, lambda_=1)))
    print("L2 Reg lambda 1, w/ dev set: RMSE: {0[0]} r2: {0[1]}".format(
        regression_scores(train_feat, train_age, dev_feat, dev_age, lambda_=1)))
    print("Length only, L2 lambda 1, w/o dev set: RMSE: {0[0]} r2: {0[1]}".format(
        regression_scores(length_feat_train, train_age, lambda_=1)))
    print("Length only, L2 lambda 1, w/ dev set: RMSE: {0[0]} r2: {0[1]}".format(
        regression_scores(length_feat_train, train_age, length_feat_dev, dev_age, lambda_=1)))
    ### E2.5 - tune and obtain the best regularization constant
    best_l = tune(train_feat, train_age, dev_feat, dev_age)
    print("Best regularization strength: {}.".format(best_l))
    print("L2 Best Tune, w/ dev set: RMSE: {0[0]} r2: {0[1]}".format(
        regression_scores(train_feat, train_age, dev_feat, dev_age, lambda_=best_l)))

    ### E2.6 - Write test predictions to 'predictions.txt'
    _, test_feats = read_data(test_file, shuffle=False)

    regr_test = linear_model.Ridge(best_l)
    regr_test.fit(feats, age)
    y_pred_test = regr_test.predict(test_feats)
    predics = np.vstack(y_pred_test)
    final_test = np.concatenate((predics, test_feats), axis=1)

    with open('a2-test.txt', 'w') as test_txt:
        np.savetxt(test_txt, final_test, fmt="%2.20f", delimiter="\t")


