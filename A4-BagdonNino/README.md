# [Assignment 4: Unsupervised Learning](https://snlp2020.github.io/a4/)

**Deadline: June 29, 2020 @08:00 CEST**

The aim of this assignment is getting familiar with unsupervised methods.
In particular, we will experiment with dimensionality reduction
with PCA and clustering using k-means.

The data set we will use for this purpose is the Twitter data
we collected during [assignment 1](https://snlp2020.github.io/a1/).
Public page and your repository contains
a [smaller sample](a4-corpus-small.tsv.gz) of the data
(in accordance with Twitter's developer policy).
This sample should be enough for the purposes of the assignment,
but if you want to see the effects of more data,
the full data set can be obtained from
[the private course repository](https://github.com/snlp2020/snlp).

As before, you are *strongly* recommended to use
the [scikit-learn library](https://scikit-learn.org) for this assignment.
You will probably need [numpy](https://numpy.org)
and [scipy](https://scipy.org/) as well.

Implement the exercises as indicated in the [template](a4.py).

## Exercises

### 4.1 Read the data  (1p)

Write the code for function `read_data()` in the template,
which simply reads the tab-separated file with two columns,
where the first column is the Twitter screen name,
and the second columns is the texts of the tweet.

### 4.2 Create "person vectors" (2p)

Implement the `encode_people()` function in the template.
This function should take the data read in Exercise 4.1 above,
and return a (sparse) vector of per-document average
word or n-gram counts for each person in the data set.

You are recommended to use scikit-learn's `CountVectorizer`
for counting the words per document.
You should write your own code to
average over these document vectors for each person.
For the purposes of this exercise,
simple bag-of-words features are sufficient.
However, you are recommended to experiment
with  higher level n-grams as well
(see below for more possible additional exercises).

This function should return a list of unique screen names,
and a numpy or scipy (sparse) array of shape `(n_names, n_words)`,
where n_words is the number of word types (or n-gram types)
in the whole corpus.

You are not required to, but recommended to experiment with the
following:

- Using word n-grams as features (n-gram combination of order one to up to 3
  should run workable on most computers)
- Removing stop words from the data
- Removing frequent or infrequent words
- Using character n-gram features 
- Experimenting with other text pre-processing options (e.g.,
  with/without case normalization)

See the steps below for potential ways to compare different outcomes.
Note, however, it is difficult to objectively measure the effects of
these steps, as we do not have any explicit 'cluster labels' for the people
in our data set.

### 4.3 Similarities between people (2p)

Implement the function `most_similar()` in the template
which should print top-n most similar and most dissimilar people
according to cosine similarity between the vectors representing them.

The output should be a sorted list of most similar people with the
similarity score printed next to them in a row.
Similarly, output should contain another list with most dissimilar people,
again with the similarity metric printed next to them.

You are not required to report to output, 
but you are strongly recommended to inspect to output
for a few people you know and reason
about the similarities based on the vectors you created
(possibly in combination with the different options suggested in 4.2).

### 4.4 Dimensionality reduction (2p)

Reduce the dimensionality of the "person vectors" using PCA
(`reduce_dimensions()` in the template).
You can use either `PCA` or `TruncatedSVD`
implementations from scikit-learn.

You are required to set the dimensions of the resulting (dense)
vectors such that it is the minimum dimensionality 
that explains at least 99% of the variance.

Experiment with similarities with the dense vectors
(you do not need to show this for grading,
but you are probably be curious if anything changes
after dimensionality reduction,
e.g., in the ranked similarities you inspected earlier).

### 4.5 Plotting the low-dimensional data (1p)

Implement the `plot()` function in the template,
which plots the screen names of people on coordinates 
of two dimensions of the (dense) vectors indicated given as arguments,
and writes the output to an external file.
The coordinate each screen name printed should be
(by default) the first and second dimensions of the corresponding dense vector.

Are the distances you see on the graph what you expect
from earlier calculations?
(Again, you do not need to answer the question or analyze this formally,
but you should inspect and compare.)

### 4.6 Clustering with k-means (1p)

Implement the function `cluster_kmeans()`,
which clusters the (dense) vectors obtained in 4.4
into pre-specified number of clusters,
and prints out the screen names of the people in each cluster.

The output should look like (for 8 clusters)
```
0: f b c g e
1: a d
...
7: h x y
```
the numbers are the cluster numbers and letters are
the screen names.

### 4.7 Picking the number of clusters (1p)

Implement the function `plot_scree()` in the template,
that plots a scree plot of k-means clusters
from `k=2` to number of clusters specified by `max_k`.
The score plotted on the scree plots should be
the silhouette score of the predicted clusters.

You should determine the number of clusters based on this graph
(you do not need to show this for grading).
