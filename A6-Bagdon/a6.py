#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 6
    See <https://snlp2020.github.io/a6/> for detailed instructions.

    Course:      Statistical Language processing - SS2020
    Assignment:  Assignment 6
    Author(s):   Chris Bagdon
    Description: Sequence labeler which finds segments from sequences
    Honor Code:  I pledge that this program represents my own work.
"""
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, RepeatVector, Dropout


def read_data(filename, eos='#'):
    """ Read an input file

    Parameters:
    -----------
    filename:  The input file
    eos:       Symbol to add to the end of each sequence (utterance)

    Returns: (a tuple)
    -----------
    utterances: A list of strings without white spaces 
    labels:     List of sequences  of 0/1 labels. '1' indicates the
                corresponding character in 'utterances' begins a word,
                '0' indicates that it is inside a word.
    """
    ### Exercise 6.1

    with open(filename) as f:
        utterances = []
        labels = []

        for line in f:
            # Get utterance output and length
            utter = line
            utter = utter.replace(" ", "").replace("\n", "") + "#"
            utterances.append(utter)
            # Make empty sequence
            sequence = np.zeros(len(utter), dtype=int)
            sequence[0], sequence[len(utter) - 1] = 1, 1
            # Find indexes of beginning of words
            prev_char = ""
            count = 0
            new_word_indexs = []
            for char in line:
                if char == " ":
                    prev_char = char
                    continue
                if prev_char == " ":
                    prev_char = char
                    new_word_indexs.append(count)
                    count += 1
                else:
                    prev_char = char
                    count += 1
            for index in new_word_indexs:
                sequence[index] = 1
            labels.append(sequence)

    return (utterances, labels)


def segment(u_train, l_train, u_test):
    """ Train an RNN sequence labeller on the training set, return segmented test set.
    Parameters:
    -----------
    u_train:  The list of unsegmented utterances for training
    l_train:  Training set labels, corresponding to each character in 'u_train'
    u_test:   Unsegmented test input

    the format of the u_train and u_test is similar to 'utterances'
    returned by 'read_data()'


    Returns: 
    pred_seg:  Predicted segments, a list of list of strings, each
               corresponding to a predicted word.
    """
    ### Exercise 6.2

    # Encode input and gather stats for model
    encoder = WordEncoder()
    encoder.fit(u_train)
    X_train = encoder.transform(u_train, flat=False)
    X_test = encoder.transform(u_test, flat=False)
    input_N = len(u_train)
    maxLen = encoder.max_wordlen
    third_dim = np.shape(X_train)[2]

    # Pad output to match input
    paddded_l_train = []
    for label in l_train:
        new_label = np.zeros(maxLen, dtype=int)
        for j, k in enumerate(label):
            new_label[j] = k
        paddded_l_train.append(new_label)
    l_train_array = np.array(paddded_l_train).reshape(input_N, maxLen, 1)

    # Build model
    model = Sequential()
    # I could not get the masking layer to work. Tried editing the X_train and Y arrays in numerous ways
    # But couldn't figure out why it wouldn't accept it. Any advice here would be much appreciated!
    # model.add(Embedding(input_dim=maxLen, output_dim=(maxLen, third_dim), mask_zero=True))
    model.add(LSTM(100, input_shape=(maxLen, third_dim), activation="relu", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    model.summary()
    model.fit(X_train, l_train_array, epochs=50, callbacks=[es])

    predictions = model.predict(X_test)
    # Convert label predictions to segments
    segments = label_to_segments(u_test, predictions)

    return segments


def evaluate(gold_seg, pred_seg):
    """ Calculate and print out boundary/word/lexicon precision recall and F1 scores.

    See the assignment sheet for definitions of the metrics.

    Parameters:
    -----------
    gold_seg:  A list of sequences of gold-standard words 
    pred_seg:  Predicted segments.

    Returns: None
    """
    ### Exercise 6.3
    BP_total_pred, BP_correct_pred, BP_gold = 0, 0, 0
    WP_total_pred, WP_correct_pred, WP_gold = 0, 0, 0
    LP_total_pred, LP_correct_pred, LP_gold = 0, 0, 0

    gold_lexicon = []
    pred_lexicon = []

    # Cycle through each utterence, tallying True positives, total predictions, and gold standards
    for gold_list, pred_list in zip(gold_seg, pred_seg):
        cur_gold_utter = []
        cur_pred_utter = []
        gold_bounds = []
        pred_bounds = []
        gold_pos = 0
        pred_pos = 0
        # Find the bounds of each segment
        for g in gold_list:
            gold_bounds.append(gold_pos)
            cur_gold_utter.append((g, gold_pos))
            gold_pos += len(g)
        for p in pred_list:
            pred_bounds.append(pred_pos)
            cur_pred_utter.append((p, pred_pos))
            pred_pos += len(p)
        # Check if bounds are  and tally
        for bound in pred_bounds:
            if bound == 0:
                continue
            if bound in gold_bounds:
                BP_correct_pred += 1
        BP_gold = BP_gold + len(gold_bounds) - 1
        BP_total_pred = BP_total_pred + len(pred_bounds) - 1
        # Check if predicted words are correct and tally. Also check if word is in lexicon
        for segment in cur_pred_utter:
            if segment[0] == "#":
                continue
            if segment in cur_gold_utter:
                WP_correct_pred += 1
                WP_total_pred += 1
                if segment[0] not in pred_lexicon:
                    LP_correct_pred += 1
                    LP_total_pred += 1
                    pred_lexicon.append(segment[0])
            else:
                WP_total_pred += 1
                if segment[0] not in pred_lexicon:
                    LP_total_pred += 1
                    pred_lexicon.append(segment[0])
        # Tally gold standard words and lexicon
        for segment in cur_gold_utter:
            if segment[0] is not "#":
                if segment[0] not in gold_lexicon:
                    gold_lexicon.append(segment[0])
                    LP_gold += 1
                WP_gold += 1
    # Calculate stats
    BP = BP_correct_pred / BP_total_pred
    BR = BP_correct_pred / BP_gold
    BF1 = 2 * ((BP * BR) / (BP + BR))

    WP = WP_correct_pred / WP_total_pred
    WR = WP_correct_pred / WP_gold
    WF1 = 2 * ((WP * WR) / (WP + WR))

    LP = LP_correct_pred / LP_total_pred
    LR = LP_correct_pred / LP_gold
    LF1 = 2 * ((LP * LR) / (LP + LR))

    print("Boundary Precision: {}\n Boundary Recall: {}\n Boundary F1: {}"
          .format(BP, BR, BF1))
    print("Word Precision: {}\n Word Recall: {}\n Word F1: {}"
          .format(WP, WR, WF1))
    print("Lexicon Precision: {}\n Lexicon Recall: {}\n Lexicon F1: {}"
          .format(LP, LR, LF1))


def label_to_segments(utters, labels):
    """Return segmented utterances corresponding (B/I) labels

        Note that to use it with model predictions, you should assume a
        number larger than 0.5 in the 'labels' array indicates a boundary.

        Parameters:
        -----------
        utters:  The list of unsegmented utterances.
        labels:      The B/O labels (probabilities) for given utterances.

        Returns:
        segment_list:  Segmented utterances.

        """
    segment_list = []
    for i, utterence in enumerate(utters):
        segments = []
        seg = ""
        for j, char in enumerate(utterence):
            if labels[i][j] >= 0.5:
                if len(seg) > 0:
                    segments.append(seg)
                seg = ""
                seg = seg + char
            else:
                seg = seg + char
            if j == (len(utterence) - 1):
                segments.append(seg)
        segment_list.append(segments)
    return segment_list


class WordEncoder:
    """An encoder for a sequence of words.

    The encoder encodes each word as a sequence of one-hot characters.
    The words that are longer than 'maxlen' is truncated, and
    the words that are shorter are padded with 0-vectors.
    A Symbol is reserved for unknown characters.

    The result is either a 2D vector, where all character vectors
    (including padding) of a word are concatenated as a single vector,
    o a 3D vector with a separate vector for each character

    Parameters:
    -----------
    maxlen:  The length that each word is padded
             or truncated. If not specified, the fit() method should
             set it to cover the longest word in the training set.
    """

    def __init__(self, maxlen=None):
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
        self.char_list.append("unknown")

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
        array = np.zeros(shape=(len(words), self.max_wordlen, num_chars), dtype=np.float32)
        for i, word in enumerate(words):
            # Use this variable to change padding location
            padding = self.max_wordlen - len(word)
            if pad == 'right':
                padding = 0
            for k, char in enumerate(word):
                # Get index of current char
                if char in self.char_list:
                    char_index = self.char_list.index(char)
                else:
                    char_index = -1
                if k > self.max_wordlen:
                    continue
                # Check if current char is in char list
                if char_index > 0:
                    array[i][padding + k][char_index] = 1
                else:
                    array[i][padding + k][num_chars - 1] = 1

        if flat:
            return array.reshape((array.shape[0], -1))
        return array


if __name__ == '__main__':
    # Approximate usage of the exercises (not tested).
    u, l = read_data('br-text.txt')
    train_size = int(0.8 * len(u))
    u_train, l_train = u[:train_size], l[:train_size]
    u_test, l_test = u[train_size:], l[train_size:]

    seg_test = segment(u_train, l_train, u_test)
    evaluate(label_to_segments(u_test, l_test), seg_test)
