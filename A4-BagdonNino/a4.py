#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 4
    See <https://snlp2020.github.io/a4/> for detailed instructions.

    Course:      Statistical Language processing - SS2020
    Assignment:  Assignment 4
    Author(s):   Chris Badgon, Nino Meisinger
    
    Honor Code:  I pledge that this program represents my own work.
"""

import csv
import gzip
import math
import numpy as np
import scipy
import sklearn
from numpy import unique, mean, array
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def read_data(fname):
    """ Read the tab-separated data.

    Parameters:
    -----------
    filename:  The name of the file to read.

    Returns: (a tuple)
    -----------
    names:      Screen names, a sequence texts with (repeated) screen
                names in the input file. The length of the sequence
                should be equal to the number of instances (Tweets) in
                the data.
    texts:      The corresponding texts (a sequence, e.g., a list)
    """
    names = []
    texts = []

    with gzip.open(fname, 'rt') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            names.append(row[0])
            texts.append(row[1])
    return (names[1:], texts[1:])


def encode_people(names, texts):
    """ Encode each person in the data as a (sparse) vector.

    The input to this function are the texts associated with screen
    names in the data. You are required to encode the texts using
    CountVectorizer (from sklearn), and take the average of all
    vectors belonging to the same person.

    You are free to use either sparse or dense matrices for the
    output.

    Parameters:
    -----------
    names       A sequence of length n_texts with (repeated) screen names
    texts       A sequence of texts

    Returns: (a tuple)
    -----------
    nameset:    (Unique) set of screen names 
    vectors:    Corresponding average word-count vectors
    """
    vectorizer = CountVectorizer()
    total_counts = vectorizer.fit_transform(texts)

    nameset = list(set(names))
    nameset.sort()
    vectors = np.empty((len(nameset), total_counts.get_shape()[1]))

    for i, user in enumerate(nameset):
        user_texts = []
        for index, name in enumerate(names):
            if name == user:
                user_texts.append(index)
        user_texts = np.array(user_texts)
        user_array = total_counts.tocsr()[user_texts,:]
        mean_vector = mean(user_array, axis=0)
        vectors[i,:] = mean_vector[0,:]

    vectors = csr_matrix(vectors)
    return (nameset, np.asarray(vectors.todense()))


def most_similar(name, names, vectors, n=10):
    """ Print out most similar and most-dissimilar screen names for a screen name.

    Based on the vectors provided, print out most similar (according to
    cosine similarity) and most dissimilar 'n' people.

    Parameters:
    -----------
    name        The screen name for which we calculate the similarities
    names       The full set of names
    vectors     The vector representations corresponding to names
                (e.g., output of encode_people() above)
    n           Number of (dis)similar screen names to print

    Returns: None
    """

    results = []
    n_vector = vectors[names.index(name)]

    # compute cosim
    for i, na in enumerate(names):
        o_n_vector = vectors[i]
        results.append((na, cosine_similarity([n_vector], [o_n_vector])[0][0]))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print("Most similar people:")
    for i in range(n+1):
        # skip first entry, which is the person itself
        if i == 0:
            continue
        print('{name:<15} {cosim}'.format(name=str(results[i][0]+":"), cosim=results[i][1]))

    print("\nLeast similar people:")
    for i in range(n):
        l = len(results)
        print('{name:<15} {cosim}'.format(name=str(results[l-i-1][0]+":"), cosim=results[l-i-1][1]))
            


def reduce_dimensions(vectors, explained_var=0.99):
    """ Reduce the dimensionality with PCA (technique, not necessarily the implementation).

    Transform 'vectors' to a lower dimensional space with PCA and return
    the lower dimensional vectors. You can use any PCA implementation,
    e.g., PCA or TruncatedSVD from sklearn (scipy also has PCA/SVD
    implementations).

    The number of dimensions to return should be determined by the
    parameter explained_var, such that the total variance
    explained by the lower dimensional representation should be the
    minimum dimension that explains at least explained_var.

    Parameters:
    -----------
    vectors      Original high-dimensional (n_names, n_features) vectors
    explaind_var The amount of variance explained by the resulting
                 low-dimensional representation.

    Returns: 
    -----------
    lowdim_vectors  Vectors of shape (n_names, n_dims) where n_dims is
                    (much) lower than original n_features.
    """

    pca = PCA(n_components=explained_var, svd_solver='full').fit(vectors)

    return pca.transform(vectors)


def plot(names, vec, xi=0, yi=1, filename='plot-2d.pdf'):
    """ Plot the names on a 2D graph.

    This function should plot the screen names (the text) at
    coordinates of vec[i][xi] and vec[i][yi] where 'i' is the index of
    the screen name being plotted.

    Parameters:
    -----------
    names       Screen names
    vectors     Corresponding vectors (n_names, n_features)
    xi,yi       The dimensions to plot
    filename    The output file name

    Returns: None
    """

    plt.figure(figsize=(25, 15))
    plt.plot(vec[:,xi], vec[:,yi], 'b.')

    for i, name in enumerate(names):
        plt.annotate(name, (vec[i][xi], vec[i][yi]))
        
    plt.savefig(filename)
    plt.close()


def cluster_kmeans(names, vectors, k=5):
    """ Cluster given data using k-means, print the resulting clusters of names.

    Parameters:
    -----------
    names       Screen names
    vectors     Corresponding vectors (n_names, n_dims)
    k           Number of clusters
    filename    The output file name

    Returns: None
    """
    kmeans = KMeans(n_clusters=k, random_state=10)
    kmeans.fit(vectors)

    labels = kmeans.labels_
    sorted_labels = [[] for i in range(k)]

    for nameID, group in enumerate(labels):
        sorted_labels[group].append(names[nameID])

    for i in range(k):
        print("{group}: {names}".format(group=i, names=" ".join(sorted_labels[i])))


def plot_scree(vectors, max_k=20, filename="scree-plot.pdf"):
    """ Plot a scree plot of silhouette score of k-means clustering.

    This function should cluster the given data multiple times from k=2
    to k=max_k, calculate the silhouette score, and plot a scree plot
    to the indicated file name.

    Parameters:
    -----------
    vectors     The data coded as (dense) vectors.
    max_k       The maximum k to try.
    filename    The output file name

    Returns: None
    """

    s_scores = []
    k_numb = []

    for i in range(2, max_k + 1):
        cluster = KMeans(n_clusters=i, random_state=10)
        cluster.fit(vectors)
        cluster_labels = cluster.predict(vectors)
        silhouette = silhouette_score(vectors, cluster_labels)

        s_scores.append(silhouette)
        k_numb.append(i)

    # set plot
    x_range = range(math.floor(min(k_numb)), math.ceil(max(k_numb)) + 1)
    plt.xticks(x_range)
    plt.ylabel("silhouette score")
    plt.xlabel("k cluster")
    plt.plot(k_numb, s_scores, linestyle='-', marker='o', color='b')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # Your main code goes below - not required for the assignment.
    names, texts = read_data("a4-corpus-small.tsv.gz")

    nameset, csr_vectors = encode_people(names, texts)
    lowdim_vectors = reduce_dimensions(csr_vectors)
    print(csr_vectors.shape)
    print(lowdim_vectors.shape)

    most_similar("TwoPaddocks", nameset, csr_vectors)    
    print("_________________________________________")
    most_similar("TwoPaddocks", nameset, lowdim_vectors)

    cluster_kmeans(nameset, lowdim_vectors)

    plot(nameset, lowdim_vectors)
    plot_scree(lowdim_vectors)

