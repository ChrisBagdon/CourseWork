#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 3
    See <https://snlp2020.github.io/a3/> for detailed instructions.

    Course:      Statistical Language processing - SS2020
    Assignment:  Assignment 3
    Author(s):   Chris Bagdon, Steinar Grässel
    Description: Classifies words into languages based on bigrams
    Honor Code:  I pledge that this program represents my own work.
"""
import gzip
import numpy as np
from numpy import genfromtxt
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


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
    words:      The list of words in the input file.
    languages:  The corresponding language codes.
    """
    with gzip.open(filename, 'rt') as f:
        # read in data
        tsv_data = genfromtxt(f, dtype="str", delimiter="\t")
        # convert to numpy array
        features = np.array(tsv_data)
        # shuffle the data if wanted
        if shuffle:
            np.random.shuffle(features)
        # slice first column
        words = features[:, 0]
        # remove it from the array
        languages = features[:, 1]
        return words, languages


def get_bigrams(word, bow='<', eow='>'):
    """ Calculate and return the character bigrams of a word.

    Parameters:
    -----------
    word:      A sequence (word) to be processed.
    bow, eow:  Beginning- and end-of-word symbols to prepend/append.
               (defaults are fine, since they do not occur in the data)

    Returns:
    -----------
    bigrams:   A sequence of (character) bigrams.
    """
    # add beginning- and end-of-word symbol to the word
    word = bow + word + eow
    bigrams = []
    # add all bigrams of the word to the list
    for index in range(len(word) - 1):
        bigrams.append(word[index:index + 2])
    return bigrams


class BigramEncoder:
    def __init__(self):
        """Initialize any data/state here if you need it. 

        You can extend the API (adding arguments as needed).
        """
        self.bigram_dict = {}
        self.bigram_list = []
        self.dummy_model = None

    def fit(self, words, overwrite=False):
        """ Calculate/update necessary bookkeeping data.

        You are free to chose your implementation. However, what you
        typically need is to build/update a mapping between bigrams
        and unique integers representing each bigram, so that for each
        word in "transform()" below, you can map a bigram to an
        integer (index) value.
        """
        # overwrite data if specified
        if overwrite:
            self.bigram_dict = {}
        # get the bigrams for all the words...
        for word in words:
            bigrams = get_bigrams(word)
            # ... and check for each bigram if it's already in our bigram dictionary...
            for bigram in bigrams:
                # ... if not add it do the dictionary and give it a unique number
                if bigram not in self.bigram_dict:
                    # will mostly use dictionary, cause it's more efficient
                    self.bigram_dict[bigram] = len(self.bigram_dict)
                    # but use a list for the inv_transform
                    self.bigram_list.append(bigram)

    def transform(self, words):
        """Transform the given words according to the already fitted encoder.
    
        Parameters:
        -----------
        words:     A list of words (a sequence of sequences/strings)

        Returns:
        -----------
        encoded_words: A numpy or scipy sparse array of shape (n, m)
                       where 'n' is the number of words, 'm' is the
                       number of unique bigrams in the data used for
                       fitting the encoder. Use of sparse matrices is
                       strongly recommended.
        """
        # construct a sparse matrix
        encoded_words = dok_matrix((len(words), len(self.bigram_dict)))
        for index, word in enumerate(words):
            bigrams = get_bigrams(word)
            for bigram in bigrams:
                if bigram in self.bigram_dict:
                    encoded_words[index, self.bigram_dict[bigram]] = 1

        return csr_matrix(encoded_words)

    def inv_transform(self, bigram_id, one_hot=False):
        """Return the bigram given its integer or one-hot representation.

        Parameters:
        -----------
        bigram_id: The integer ID/representation of a bigram.
        one-hot:   If True, bigram_id is a one-hot representation where
                   only the element corresponding to the bigram index is 1.

        Returns:
        -----------
        The bigram or None if bigram is unknown.
        """
        if bigram_id < len(self.bigram_list):
            return self.bigram_list[bigram_id]
        else:
            return None


def cross_validate(model_type, words, languages, encoder=None, k=3, verbose=False):
    """Select and return the best model using cross-validation.

    The parameters and the range should be chosen based on the 'model'
    parameter. At a minimum you should tune the parameter 'C' (inverse
    of the regularization strength) for logistic regression, and
    'alpha' (Laplace smoothing factor) for naive Bayes. You may, but
    not required to, tune any other hyperparameter that may be important.

    The best model should be selected based 'macro averaged' F1 score.

    Parameters:
    -----------
    model_type:'nb' for 'naive Bayes' or 'lr' for logistic regression.
               You are support other classifiers here.
    words:     The list of words for training/validation.
    languages: The corresponding labels (languages).
    encoder:   If None, you should fit an encoder object here.
               This is included in case you want to tune some aspects
               of the encoder as well. If not, you should just pass an
               encoder fitted for all models you tune.
    k:         The number of folds
    verbose:   If true, print out state/results of the tuning process.

    Returns:
    -----------
    A tuple with a classifier and an encoder object
    """
    if verbose is True:
        verbose = 10
    if encoder is None:
        encoder = BigramEncoder()
        encoder.fit(words)

    matrix = encoder.transform(words)
    x_train, x_test, y_train, y_test = train_test_split(matrix, languages, test_size=0.4, random_state=0)
    if model_type == "nb":
        clf = GridSearchCV(BernoulliNB(), {"alpha": np.power(10, np.arange(-4, 1, dtype=float))}, scoring="f1_macro",
                           cv=k, verbose=verbose)
    elif model_type == "lr":
        clf = GridSearchCV(LogisticRegression(dual=False, solver="saga", max_iter=10000),
                           {"C": np.power(10, np.arange(-4, 1, dtype=float))},
                           scoring="f1_macro", cv=k, verbose=verbose)
    else:
        print("Invalid model type. Use either \"nb\" or \"lr\"")
        return None

    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_true, y_pred = y_test, clf.predict(x_test)
    print(clf.best_estimator_)
    return clf.best_estimator_, encoder


def evaluate(models, encoder, words, languages):
    """Print out evaluation metrics.
    
    This function should print out evaluation metrics / analyses
    described in the assignment sheet (Exercise 3.3).


    Parameters:
    -----------
    models:    A sequence of classifier objects.
    encoder:   The feature encoder (optionally as sequence if you use
               different encoders for different classifiers).
               We assume that you have a common encoder for all models
               you evaluate. If you use different encoders, feel free
               to allow a sequence here.
    words:     Test words.
    languages: The corresponding labels (languages)...

    Returns:
    -----------
    None
    """
    matrix = encoder.transform(words)
    for model in models:
        y_true, y_pred = languages, model.predict(matrix)
        print(classification_report(y_true, y_pred))
        unique_languages = list(set(y_true))
        length = len(unique_languages)
        conf_matrix = np.array(confusion_matrix(y_true, y_pred, unique_languages))
        np.fill_diagonal(conf_matrix, 0)
        k = 0
        for i in range(length):
            k += 1
            for j in range(k, length):
                conf_matrix[i, j] = conf_matrix[i, j] + conf_matrix[j, i]
                conf_matrix[j, i] = 0

        for i in range(10):
            index = np.unravel_index(np.argmax(conf_matrix), (length, length))
            print(unique_languages[int(index[0])] + ":" + unique_languages[int(index[1])] + " = "
                  + str(conf_matrix[int(index[0]), int(index[1])]))
            conf_matrix[int(index[0]), int(index[1])] = 0

    x_train, x_test, y_train, y_test = train_test_split(matrix, languages, test_size=0.4, random_state=0)
    dummy_classifier = DummyClassifier(strategy='most_frequent', random_state=0)
    dummy_classifier.fit(x_train, y_train)
    DummyClassifier(random_state=0, strategy='most_frequent')
    print("The random baseline score is:")
    print(dummy_classifier.score(x_test, y_test))


def print_predictors(model, encoder, language, model_type, n=10):
    """Print out best predictors for the given language.
    
    This predictors (bigrams) that are most useful for the given
    language.

    Parameters:
    -----------
    model:      A classifier object
    encoder:    The feature encoder (optionally as sequence if you use
                different encoders for different classifiers).
    model_type: A string describing the classifier, e.g., 'nb', or 'lr'.
    language:   The 3-letter language code.
    n:          Number of 'top' predictors to print.

    Returns:
    -----------
    None
    """
    language_index = np.where(model.classes_ == language)[0][0]

    if model_type == "lr":
        predictivity = np.abs(model.coef_)

    elif model_type == "nb":
        predictivity = model.feature_log_prob_

    else:
        print("Invalid model type. Exiting method")
        return

    predictivity = predictivity[language_index, :].argsort()
    best_predictors = predictivity[-n:]
    # print the top 'n' predictors
    for index in range(10):
        print(encoder.inv_transform(best_predictors[index]))


if __name__ == '__main__':
    words, languages = read_data("a3-train.tsv.gz")
    models = [cross_validate("nb", words, languages, verbose=True)[0]]
    bigram_encoder = BigramEncoder()
    bigram_encoder.fit(words)
    words, languages = read_data("a3-test.tsv.gz")
    evaluate(models, bigram_encoder, words, languages)
    # using japanese for obvious results
    print_predictors(models[0], bigram_encoder, "jpn", "nb")


