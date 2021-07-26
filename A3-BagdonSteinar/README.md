# [Assignment 3: Classification, cross validation](https://snlp2020.github.io/a3/)

**Deadline: June 15, 2020 @08:00 CEST**

This assignment includes exercises with 'traditional' classification
methods and cross validation.
We will work on these methods in a real (but simplified) problem:
language detection.
In particular,
we will build models to predict language from only a single word.
Our data set consist of a simple tab-separated file format
where the first column is a word
(with considerable noise,
since determining which tokens make up a word is not trivial for all
languages in the data),
and the second column is the 
[ISO 639-2](https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes)
code of the one of 173 languages in the data set.
The data comes from the
[Leipzig Corpora Collection](https://corpora.uni-leipzig.de/)
(Goldhahn, Eckart and Quasthoff, [2012](http://www.lrec-conf.org/proceedings/lrec2012/pdf/327_Paper.pdf)).

The data for this assignment is included in your repository as
[a3-train.tsv.gz](a3-train.tsv.gz) and
[a3-test.tsv.gz](a3-test.tsv.gz).
The data files do not have a header.

Please implement all exercises as indicated in the provided [template](./a3.py).
Use Python [scikit-learn](https://scikit-learn.org/) library
for the classification models.
You will also need to make use of [numpy](https://numpy.org/)
and/or [scipy](https://www.scipy.org/)
in some of the exercises.

## Exercises

### 3.1 Read and encode the data (3p)

Implement the functions/methods for reading the data
and encoding the words as a set of character bigram features.
For each word, also prepend/append a beginning-of-word
and an end-of-word symbol.
Given the word '_word_' the features should be
`['<w', 'wo', 'or', 'rd', 'd>']`.
And, once encoded, each feature should map to a unique index,
and each word should be represented by a binary vector,
where indices corresponding to its features are 1,
and all others values are 0.
For example, assuming the indices of the bigrams above are
1, 3, 5, 7, 10 respectively, the corresponding binary vector should
look like the following.

```
0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, ...
```

Note that we are using binary 'indicator' features,
any number of occurrences of a bigram is mapped to a 1
regardless of how many times it occurs.
Please follow the instructions in the template file for
the details of the interface.
For the purposes of this assignment,
implementing only a bigram feature encoder is sufficient.
However, more flexible encoders are also welcome.

Do **not** use a high-level library utility (e.g.,
`CountVectorizer` from sklearn) for this task.

### 3.2 Classification and cross validation (3p)

Implement the function `cross_validate()` in the template,
which tunes the hyperparameters of a given model.
Your function should support tuning parameters of
logistic regression and naive Bayes classifiers as
implemented by [sklearn](https://scikit-learn.org/).

You are encouraged, but not required to, support other classifiers
from sklearn. 

For this exercise you can use any of the utilities provided by the
[sklearn.model_selection]( https://scikit-learn.org/stable/model_selection.html).

### 3.3 Classifier evaluation (3p)

Implement the function `evaluate()` which prints out
the macro-averaged precision, recall and F1-scores
for each classifier you tuned in exercise 3.2,
and 'most confused' 10 language pairs by each classifier,
and a random baseline on the test data.

We define 'most confused' as the language pairs with highest
number of confusions in both directions.
That is, the pairs `(a, b)`
where the sum of misclassified instances of `a` as `b`
and misclassified instances of `b` as `a` is the highest.

### 3.4 Inspect the predictors (1p)

Implement the function `print_predictors()` which prints out
the most important/useful predictors for a given language
for both naive Bayes and logistic regression classifiers
you obtained in exercise 3.2.
