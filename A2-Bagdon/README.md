# [Assignment 2: Correlation, regression, regularization](...)

**Deadline: May 22, 2020 @12:00 CEST**

In this set of exercises, we will experiment with linear regression
and regularization.
The data comes from a corpus of poems by children 
[(Hipson and Mohammad, 2020)](https://arxiv.org/abs/2004.06188v3),
but the version we will work on
has already been pre-processed,
and provided as a set of numeric features,
rather than text instances.
In this exercise we will try to predict age of the author
from the features extracted from texts (poems) written by them.

You can find the data in your repository as two separate files.
The file [train.tsv.gz](train.tsv.gz) contains training instances,
including the gold-standard values for `age`
while the data in [test.tsv.gz](test.tsv.gz) contains test instances.
Data is formatted as tab-separated files in both cases.
First two columns contain `age` and document `length`,
while the rest of the columns are numeric features extracted from each document
based on words and characters they contain
(we will see how it is done later in this course).
The age column in the test file is arbitrarily set to 0
in the test set.

Use Python [scikit-learn](https://scikit-learn.org/) library
for the regression models.
You will also need to make use of [numpy](https://numpy.org/)
in some of the exercises.

Please implement your solutions as instructed in the provided
[template](a2.py).

## Exercises

### 2.1 Read the data (1p)

Implement the function `read_data()` that reads the data files
as indicated in the provided [template](a2.py).
Read the file `train.tsv.gz`,
shuffle the data (in each row), and split the data into two sets:
'training' (80%) and 'development', or 'validation' (20%).

### 2.2 Calculate the correlation between two variables (2p)

Implement the function `correlate` in the template.
For this exercise, **do not** use any high-level library functions.
You are only allowed to use Python built-in `sum()` functions
and arithmetic operations.

Check the correlation between `age` and document `length`,
and compare your results with the `numpy.corrcoef()`.
The results should be the same (with some rounding error).

You are also recommended to inspect the correlation between 
other variables,
both the correlation between age and the other predictors,
and between different predictors.

### 2.3 Fit a least-squares regression model (2p)

Fit a least-squares regression model (without any regularization term)
to the data as indicated in the template file.
You should fit your model on the training set only.
Compare the r-squared and RMSE on the training
and the development set.

Furthermore, compare the model performance
using only the document length as a feature,
and the whole set of predictors (`length` and `xNNN`).

### 2.4 Add a regularization term, compare (2p)

Repeat Exercise 2.3 above with a model with L2 regularization.
Compare your results with the non-regularized model in 2.3.
For this exercise, you can use the 1.0 (also library default)
as regularization constant.

### 2.5 Find a good regularization coefficient (2p)

Implement the function `tune` in the template,
which trains and tests the model in 2.4 
using different values of the regularization constant,
and returns the optimum value.

You are free to implement your own strategy
for the search for the best regularization constant.
**Do not** use external library functions,
e.g., from `sklearn.model_selection` for this exercise.

### 2.6 Predict ages for test instances (1p + 1p)

Using your best model settings,
predict the age values for the given test instances
in [test.tsv.gz]test.tsv.gz),
and write to a file named `predictions.txt`.
Each line the output file `predictions.txt` should contain
prediction of age the age of the author with the given features.
Make sure you preserve the order of the test instances
while writing the labels (the order of instances in
`test.tsv.gz` and `predictions.txt` should match.
**Do not** include a header row in `predictions.txt`.
Remember that the age column in `test.tsv.gz` is just a placeholder,
you do not have access to the actual age values of the test instances.

You can use the best model you found in exercise 2.5.
However, you are also welcome to try different models/options
to get the best score (e.g., trying a non-linear regression model).

The scoring for this exercise involves a bonus point.
You will get the full score from the exercise if
your model scores better than a trivial baseline
(always offering the mean age of the gold-standard test instances
as the predicted age).
The teams with top 5 r-squared values on the test set
will get a bonus point.
