# [Assignment 6: Segmentation as Sequence labeling](https://snlp2020.github.io/a6/)

**Deadline: July 27, 2020 @08:00 CEST**

In this assignment we will work with (gated) recurrent networks
for segmentation.
In particular, we treat the segmentation task as a sequence labeling task,
where we want to label each character as 'beginning of a word' (B, or 1)
or 'inside a word' (I, or 0).
This is very similar to BIO tagging used for many problems in NLP,
e.g., tokenization or named entity recognition.
In our case, we assume that there will be no 'other' (O) tag
(for example space characters for tokenization,
or words that do not belong to any named entity).

In this particular form,
the problem is relevant for segmentation of languages
where the written text does not include word boundaries,
or processing spoken language transcripts without (clear) word
boundary information.
Although you can work with any data set you like,
for your convenience (and comparison of success of your systems)
your repository contains a data set which has been used
in many studies on child language acquisition.
In particular the data comes from the
[CHILDES](https://childes.talkbank.org/)
(originally collected by [Bernstein-Ratner
(1987)](https://books.google.de/books?id=fBapP38RhwsC&lpg=PA159&ots=48MF3EUK6w&dq=The%20phonology%20of%20parent-child%20speech&lr&pg=PA159#v=onepage&q&f=false)
 processed by [Brent & Cartwright (1996)](https://www.sciencedirect.com/science/article/pii/S0010027796007196)
 into to present form).
The data comes as [phonemic transcriptions](br-phono.txt)
and [regular text](br-text.txt).
In child language acquisition research,
the phonemic transcriptions are used.
You are free to experiment with any,
try both and compare the performance of your system on different data sets.

Each line in the data files contains an utterance,
and the word boundaries are indicated with space characters.

As usual, implement your system in the provided [template](a6.py).
You are expected to use
[Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
for defining the neural network in this assignment.

## Problem definition

### 6.1 Reading the data

Not surprisingly, our assignment starts with reading the data file.
Implement the function `read_data()`
that reads a data file formatted as described,
and returns utterances without spaces,
as well as `0` and `1` labels corresponding to 'inside a word'
and 'beginning of a word' respectively.
Also include a special 'end-of-utterance' symbol (`#`)
which is treated as a word.

For example, for the input file
```
night night
daddy
a kitty
```
the function should return unsegmented input like
```
["nightnight#", "daddy#", "akitty#"]
```
and  labels like
```
[[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
 [1, 0, 0, 0, 0, 1], 
 [1, 1, 0, 0, 0, 0, 1]]
```

### 6.2 RNN sequence labeller

Implement the function `segment()` in the template
that trains a (gated) RNN for predicting the boundaries
(B/I or 1/0 tags) using the unsegmented
input returned by `read_data()` described above.
The function should return sequences of segmented test utterances.

For the input
```
["nightnight#", "daddy#", "akitty#"]
```

The output should be predicted segmentations similar to:
```
[["night", "night", "#"], ["daddy#"], ["ak", "it", "ty", "#"]]
```
Note that we assume the model made some mistakes
in the example above.

You are free to chose the architecture and the data encoding
(you may want to reuse your `WordEncoder`
from the [assignment 5](https://snlp2020.github.io/a5/) 
if you opt for one-hot encoding for input characters)
and whether and how to train/tune your network.

You are not required to demonstrate your tuning.
However, you should be careful to follow correct practices.

### 6.3 Evaluating the output

Implement the function `evaluate()` in the template that takes 
gold-standard segmentations and predicted segmentations,
and produces the following metrics:

- _Boundary_ precision (BP), recall (BR) and F1 score: these scores should
    indicate the ratio of the predicted boundaries that are correct
    according to the gold standard (precision),
    the ratio of the gold-standard boundaries
    that were correctly predicted (recall),
    and their harmonic mean (F1 score).
- _Word_ precision (WP), recall (WR) and F1 score: similar to above, but to
    count a true positive, both boundaries of a word should be
    identified correctly.
- _Lexicon_ precision (LP), recall (LR) and F1 score:
    similar to word precision, but each word type (unique word)
    counts only once.

For example, for predicted segmentation
```
[["night", "night", "#"], ["daddy#"], ["ak", "itty", "#"]]
```
and the gold standard
```
[["night", "night", "#"], ["daddy", "#"], ["a" "kitty", "#"]]
```

BP = (2+0+1)/4, BR=(2+0+1)/5, WP = (2+0+0)/5, WR=(2+0+0)/5, LP = (1+0+0)/4, WR=(1+0+0)/4.

Our treatment of 'end-of-sequence' symbol may seem somewhat arbitrary.
The general guiding principle is that we do not want to credit the
system for predicting the obvious (so, a boundary before the first
word or after `#` should not add to true positives),
but predicting the end of the last word
(a 'beginning-of-word' label on `#`)
is part of the evaluation.

