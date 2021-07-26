#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 5
    See <https://snlp2020.github.io/a5/> for detailed instructions.

    Course:      Statistical Language processing - SS2020
    Assignment:  Assignment 3
    Author(s):   Chris Bagdon
    Description: POS Tagger from chars using Keras MLP and RNN
    Honor Code:  I pledge that this program represents my own work.
"""
import csv

import numpy as np
from keras import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn import preprocessing
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, TimeDistributed, Activation
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, \
    multilabel_confusion_matrix


def read_data(treebank, shuffle=True, lowercase=True,
        tags=None):
    """ Read a CoNLL-U formatted treebank, return words and POS tags.

    Parameters:
    -----------
    treebank:  The path to the treebank (individual file).
    shuffle:   If True, shuffle the word-POS pairs before returning
    lowercase: Convert words (tokens) to lowercase
    tags:      If not 'None', return only the pairs where POS tags in tags

    Returns: (a tuple)
    -----------
    words:      A list of words.
    pos:        Corresponding POS tags.
    """
    words_dict = {}
    words = []
    pos = []
    skip_count = 0
    with open(treebank, encoding='utf8') as f:
        for row in f:
            if row.startswith("#"):
                continue
            split_row = row.split("\t")
            #Skipping multiwords
            if skip_count > 0:
                skip_count = skip_count - 1
                continue
            #Skip non entries
            if len(split_row) < 5:
                continue
            #Skip empty nodes
            if "." in split_row[0]:
                continue
            #Skip Multiwords
            if "-" in split_row[0]:
                dash = split_row[0].index("-")
                skip_count = int(split_row[0][:dash] - int(split_row[0][dash:]))
                continue
            word = split_row[2]
            # This cuts out non words
            #if len(word) > 30:
                #continue
            cur_pos = split_row[4]
            #Skip words not in tags
            if tags is not None:
                if cur_pos in tags:
                    continue
            if lowercase:
                word = word.lower()
            if word in words_dict.keys():
                if cur_pos in words_dict[word]:
                    continue
                else:
                    words_dict[word].append(cur_pos)
            else:
                words_dict[word] = [cur_pos]
            #words.append(word)
            #pos.append(cur_pos)
    for k, v in words_dict.items():
        for x in v:
            words.append(k)
            pos.append(x)

    if shuffle:
        shuffle_me = np.column_stack((words, pos))
        np.random.shuffle(shuffle_me)
        words = shuffle_me[:,[0]].tolist()
        pos = shuffle_me[:,[1]].tolist()

    return (words, pos)



class WordEncoder:
    """An encoder for a sequence of words.

    The encoder encodes each word as a sequence of one-hot characters.
    The words that are longer than 'maxlen' is truncated, and
    the words that are shorter are padded with 0-vectors.
    Two special symbols, <s> and </s> (beginning of sequence and
    end of sequence) should be prepended and appended to every word
    before padding or truncation. You should also reserve a symbol for
    unknown characters (distinct from the padding).
    
    The result is either a 2D vector, where all character vectors
    (including padding) of a word are concatenated as a single vector,
    o a 3D vector with a separate vector for each character (see
    the description of 'transform' below and the assignment sheet
    for more detail.

    Parameters:
    -----------
    maxlen:  The length that each word (including <s> and </s>) is padded
             or truncated. If not specified, the fit() method should
             set it to cover the longest word in the training set. 
    """
    def __init__(self, maxlen = None):
        ### part of 5.2
        self.max_wordlen = 0
        self.char_list = []
        self.maxlen = maxlen

    def fit(self, words):
        """Fit the encoder using words.

        All collection of information/statistics from the training set
        should be done here.

        Parameters:
        -----------
        words:  The input words used for training.

        Returns: None
        """
        ### part of 5.2
        chars = set()
        for word in words:
            if len(word) > self.max_wordlen:
                self.max_wordlen = len(word)
            chars.update(word)
        if self.maxlen is not None:
            self.max_wordlen = self.maxlen
        self.char_list = list(chars)
        self.char_list.extend(["<s>", "</s>", "unknown"])


    def transform(self, words, pad='right', flat=True):
        """ Transform a sequence of words to a sequence of one-hot vectors.

        Transform each character in each word to its one-hot representation,
        combine them into a larger sequence, and return.

        The returned sequences formatted as a numpy array with shape 
        (n_words, max_wordlen * n_chars) if argument 'flat' is true,
        (n_words, max_wordlen, n_chars) otherwise. In both cases
        n_words is the number of words, max_wordlen is the maximum
        word length either set in the constructor, or determined in
        fit(), and n_chars is the number of unique characters.

        Parameters:
        -----------
        words:  The input words to encode
        pad:    Padding direction, either 'right' or 'left'
        flat:   Whether to return a 3D array or a 2D array (see above
                for explanation)

        Returns: (a tuple)
        -----------
        encoded_data:  encoding the input words (a 2D or 3D numpy array)
        """
        ### part of 5.2
        num_chars = len(self.char_list)
        return_list = []
        array = np.zeros(shape=(len(words), self.max_wordlen + 2, num_chars))
        for i, word in enumerate(words):
            vector = np.zeros(shape=(self.max_wordlen+2, num_chars))
            # Use this variable to change padding location
            padding = self.max_wordlen + 2 - len(word)
            if pad == 'right':
                padding = 0
            for k, char in enumerate(word):
                # Get index of current char
                if char in self.char_list:
                    char_index = self.char_list.index(char)
                else:
                    char_index = -1
                if k > self.max_wordlen + 2:
                    continue
                # Check if current char is in char list
                if char_index > 0:
                    array[i][padding+k+1][char_index] = 1
                    #vector[padding+k+1][char_index] = 1
                else:
                    array[i][padding+k+1][num_chars-1] = 1
                    #vector[padding + k + 1][num_chars-1] = 1
            # Mark start and end chars
            array[i][padding][num_chars - 3] = 1
            array[i][padding+len(word)+1][num_chars - 2] = 1
            #vector[padding][num_chars - 3] = 1
            #vector[padding+len(word)+1][num_chars - 2] = 1
            #return_list.append(vector)

        #array = np.array(return_list)
        if flat:
            return array.reshape((array.shape[0], -1))
        return array




def train_test_mlp(train_x, train_pos, test_x, test_pos):
    """Train and test MLP model predicting POS tags from given encoded words.

    Parameters:
    -----------
    train_x:    A sequence of words encoded as described above
                (a 2D numpy array)
    train_pos:  The list of list of POS tags corresponding to each row
                of train_x.
    test_x, test_pos: As train_x, train_pos, for the test set.

    Returns: None
    """
    ### 5.3 - you may want to implement parts of the solution
    ###       in other functions so that you share the common
    ###       code between 5.3 and 5.4

    dim = len(train_x[0])
    classes = len(set(pos))

    le = preprocessing.LabelEncoder()
    le.fit(train_pos)
    y_train = le.transform(train_pos)
    y_train = to_categorical(y_train)
    y_test = le.transform(test_pos)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(64, input_dim=dim, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    history = model.fit(train_x, y_train, epochs=5, batch_size=100, verbose=1, validation_split=0.2, callbacks=[es])
    epochs = len(history.history['loss'])

    second_model = Sequential()
    second_model.add(Dense(64, input_dim=dim, activation='relu'))
    #second_model.add(Dense(64, activation='relu'))
    second_model.add(Dense(classes, activation='softmax'))
    second_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, y_train, epochs=epochs, batch_size=100, verbose=1)

    y_classes = second_model.predict(test_x, verbose=0)
    y_pred = np.array([[1 if i > 0.5 else 0 for i in j] for j in y_classes])

    """I ran out of time to get the scores running correctly. I'm not sure how I was suppossed to handle this part.
    Using the built in tools, they want either 1 or 0, not continious values so I rounded the scores. But all scores 
    are very close to zero, or at least not where near 0.5, so all my averages are 0s."""
    print(classification_report(y_test, y_pred))
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    con_mat = multilabel_confusion_matrix(y_test, y_pred)

    print('Precision score: {}'.format(precision))
    print('Recall score: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print(con_mat)



def train_test_rnn(trn_x, trn_pos, tst_x, tst_pos):
    """Train and test RNN model predicting POS tags from given encoded words.

    Parameters:
    -----------
    train_x:    A sequence of words encoded as described above
                (a 3D numpy array)
    train_pos:  The list of list of POS tags corresponding to each row
                of train_x.
    test_x, test_pos: As train_x, train_pos, for the test set.

    Returns: None
    """
    ### 5.4
    dim = len(trn_x[0])
    classes = len(set(pos))
    third_dim = np.shape(trn_x)[2]
    le = preprocessing.LabelEncoder()
    le.fit(trn_pos)
    y_train = le.transform(trn_pos)
    y_train = to_categorical(y_train)
    y_test = le.transform(tst_pos)
    y_test = to_categorical(y_test)

    model_rnn = Sequential()

    """Also ran out of time to get my rnn fully working
    """
    #model_rnn.add(Embedding(classes, 100, input_length=30))
    model_rnn.add(LSTM(64, input_shape=(dim, third_dim), return_sequences=False))
    model_rnn.add(Dense(classes, activation='softmax'))
    #model_rnn.add(Activation('softmax'))
    model_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

    history = model_rnn.fit(trn_x, y_train, epochs=5, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[es])
    epochs = len(history.history['loss'])

    second_model = Sequential()
    #second_model.add(Embedding(classes, 100, input_length=30))
    second_model.add(LSTM(64, input_shape=(dim, third_dim), return_sequences=False))
    second_model.add(Dense(classes, activation='softmax'))
    #second_model.add(Activation('softmax'))
    second_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    second_model.fit(trn_x, y_train, epochs=epochs, batch_size=1000, verbose=1)

    y_classes = second_model.predict(tst_x, verbose=0)
    y_pred = np.array([[1 if i > 0.5 else 0 for i in j] for j in y_classes])


    print(classification_report(y_test, y_pred))
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    con_mat = multilabel_confusion_matrix(y_test, y_pred)

    print('Precision score: {}'.format(precision))
    print('Recall score: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print(con_mat)


def encode_labels(pos):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(pos)

if __name__ == '__main__':
    ### Not checked for grading,
    ### but remember that before calling train_test_mlp() and 
    ### train_test_rnn(), you need to split the as training and test
    ### set properly.
    words, pos = read_data("en_ewt-ud-train.conllu", shuffle=False)
    encoder = WordEncoder()
    encoder.fit(words)
    train_array = encoder.transform(words, flat=False)
    train_array_flat = encoder.transform(words, flat=True)
    test_words, test_pos = read_data("en_ewt-ud-test.conllu")
    test_array = encoder.transform(test_words, flat=False)
    test_array_flat = encoder.transform(test_words, flat=True)
    train_test_mlp(train_array_flat, pos, test_array_flat, test_pos)
    train_test_rnn(train_array, pos, test_array,test_pos)

