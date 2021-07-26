# [Assignment 5: Exercises with ANNs](https://snlp2020.github.io/a5/)

**Deadline: July 13, 2020 @08:00 CEST**

This assignment includes a set of exercises with artificial neural networks.
In particular, we will experiment with feed-forward (MLP)
and recurrent network architectures.

The problem we are trying to solve is a simplified POS tagging problem.
We are interested in predicting the POS tag of 
a word using only the characters (or character sequences) present in the word.
In a real-world POS tagging application,
it is almost unthinkable not to use the context of the word.
However, the way we formulate the problem 
keeps the computation requirements low and
allows for additional experimentation.

For this set of exercises the data comes from
[Universal Dependencies](https://universaldependencies.org/) treebanks.
The data is not included in your repository,
and you are free to work on any language you like.
However, your code should run on any valid
[CoNLL-U](https://universaldependencies.org/format.html)-formatted treebank.
You are expected to use
[Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
for defining the neural networks in this exercise.
As usual, please implement all exercises as indicated in the
provided [template](a5.py).

## Exercises

### 5.1 Read the data (1p)

Implement the function `read_data()` in the template,
which reads a CoNLL-U-formatted treebank, and returns unique word-POS tag pairs,
with some optional pre-processing and filtering.
In particular:

- We are interested in only word forms (column 2)
    and the coarse "universal" POS tags (column 4).
- Skip words (and their POS tags) that are part of a
    multi-word token and empty nodes
    (see [CoNNL-U documentation](https://universaldependencies.org/format.html)
    for definitions of multi-word tokens and empty node).
- If `lowercase` argument is `True`, convert all words to lowercase.
- If `shuffle` argument is `True`, shuffle the (word, POS) pairs.
- If a list of `tags` is given, return only the words (and their POS
    tags) if the POS tag is in the given list.

Note that we are working on unique word-POS combinations.
For example, for English, the data you return should have 
`the`-`DET` pair only once.

Our usage does not require any complicated processing of CoNLL-U files.
You can treat the input as tab-separated files
(after skipping comments and blank lines),
and read them without using any special library.
However, you can also to use an external library if you prefer.

### 5.2 Encoding words (3p)

Implement the class `WordEncoder` which is used for encoding a set of
words as a sequence of one-hot representations of characters in each word.
Similar to earlier assginemnts, the API defines a `fit()` method which 
collects information from a set of words (training set),
and a `transform()` method that encodes the given list of words
based on the information collected in the `fit()` method.
Please follow the instructions in the [template](a5.py) for details of the API.

The `transform()` method is required to output two related but different 
encodings for a given word.
Assume we have the following one-hot codes for letters a, b, and c:
```
a [0,0,0,0,1,0,0]
b [0,0,0,1,0,0,0]
c [0,0,1,0,0,0,0]
```
and we want to encode the words 'bb' and  'acc',
and the maximum word length is set to 4
(You are also asked to append beginning- and end-of-sequence symbols.
We skip them here for simplicity).

-  If the argument `flat` is `False`
    the output for this word should be
```
[[[0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0],
  [0,0,0,1,0,0,0],
  [0,0,0,1,0,0,0]]
 [[0,0,0,0,0,0,0],
  [0,0,0,0,1,0,0],
  [0,0,1,0,0,0,0],
  [0,0,1,0,0,0,0]]]
```
    The output should have the shape
    `(n_words, max_length, n_chars)`, where `n_words` is the number of
    input words (2 in this example),
    max_length  is the maximum word length (4 in this example)
    either set in the constructor, or determined during `fit()`,
    and `n_chars` is the number of unique characters including
    the special symbols defined in the template (7 in this example).
    This representation will be useful for training RNNs.
-  If the argument `flat` is `True`
    the output is
```
[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0]]
```
    The shape should be
    `(n_words, max_length * n_chars)`.
    This representation will be useful for training feed-forward
    networks.

### 5.3 Training and testing an MLP (4p)

Implement the function `train_test_mlp()`, which trains a simple MLP
predicting the POS tags from the encoded words.
Your function should
- train an MLP model with a single hidden layer
    with 64 units using 'relu' activation
- use part of the training set as validation set for 50 epochs
- pick the epoch with the best (smallest) validation loss,
- re-train the model (from scratch) using the full training set
    up to the best epoch determined earlier,
- and print out macro-averaged precision, recall, F1 scores,
    and the confusion matrix on the test set.

Note that the dimensions of input,
and the type/size of the final output (classification)
layer is determined by the data (and the problem).
You can freely choose the options that are not specified above,
e.g., mini batch size, or optimization algorithm,
or use the library defaults when applicable.

For computational efficiency the above procedure suggests performing
a 'naive' early-stopping method for determining when to stop (best epoch).
However, you are encouraged to experiment with tuning other hyperparameters,
for example:

- Number of units in the hidden layer
- Number of hidden layers 
- The activation function
- The batch size during training
- The optimization algorithm, and parameters of the optimization
  algorithm (e.g., learning rate)
- Dropout with different rates before and after the hidden layer(s)
- Padding direction

The model used in this exercise is simply 'wrong'
for processing sequences.
You should think about why this model is not suitable for the task,
and why it works as much as it works on this particular problem.

### 5.4 Training and testing an RNN (2p)

Implement the function `train_test_rnn()`, which trains
a gated recurrent neural network for solving the same problem.
Use a gated recurrent network of your choice (e.g., GRU or LSTM),
with 64 hidden units. The classifier layer should be trained
on the final representation built by the RNN for the whole sequence
(this is the default behaviour in Keras' RNN layers).

Your function should perform the same steps as in Exercise 5.3.
However, you should use an RNN instead of an MLP.

As well as the additional questions/tasks listed for Exercise 5.3,
you are encouraged to experiment with
different RNN architectures, including the simple RNNs,
and bidirectional versions of them.
