"""
Copyright © 2020 Johnson & Johnson
"""

import warnings
from sklearn.metrics import pairwise_distances

supported_metrics = ['cosine', 'jaccard', 'manhattan', 'dice', 'hamming']

sparse_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
dense_metrics = ['braycurtis', 'canberra', 'chebyshev', 'correlation',
                 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']


def similarity_calculation(new_mat,
                           old_mat,
                           metric: str = 'cosine'):
    """
    Calculate similarity between two sparse document-feature matrices
    representing the new document-feature matrix (1 x f)
    and the old document-feature matrix(d x f)

    :param new_mat: scipy csr object of dimensions d x f for d documents and
        f features representing the new document-feature matrix, it should
        always be 1 x f.
    :param old_mat: scipy csr object of dimensions d x f for d documents and
        f features representing the old document-feature matrix, it should
        always be d x f.
    :param metric: string indicating the similarity/distance metric to be used,
        cosine is the default.

    :return: ndarray, similarity for each old document.
    """

    # Check that metric is supported
    assert metric in supported_metrics

    # Check dimensionality of new and old are compatible
    assert new_mat.shape[1] == old_mat.shape[1]

    # Check that new only contains a single nc
    assert new_mat.shape[0] == 1

    if metric in dense_metrics:
        warnings.warn("Your choice of distance does not support sparse " + \
                      "input and will now be converted to dense " + \
                      "representation. This will take up significantly " + \
                      "more memory.")
        old_mat = old_mat.toarray()
        new_mat = new_mat.toarray()

        if metric in ['jaccard', 'dice']:
            old_mat = old_mat.astype(bool)
            new_mat = new_mat.astype(bool)

    # Calculate distance using sklearn and do 1 - distances
    similarities = 1 - pairwise_distances(new_mat, old_mat, metric=metric)

    return similarities
