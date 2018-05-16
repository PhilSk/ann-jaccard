from __future__ import absolute_import
from scipy.spatial.distance import pdist as scipy_pdist
from sklearn.metrics import jaccard_similarity_score


def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]


# Need own implementation of jaccard because numpy's implementation is different

metrics = {
    'hamming': {
        'distance': lambda a, b: pdist(a, b, "hamming"),
        'distance_valid': lambda a: True
    },
    # return 1 - jaccard similarity, because smaller distances are better.
    'jaccard': {
        'distance': lambda a, b: jaccard_similarity_score(a, b),
        'distance_valid': lambda a: True
    },
    'euclidean': {
        'distance': lambda a, b: pdist(a, b, "euclidean"),
        'distance_valid': lambda a: True
    },
    'angular': {
        'distance': lambda a, b: pdist(a, b, "cosine"),
        'distance_valid': lambda a: True
    }
}
