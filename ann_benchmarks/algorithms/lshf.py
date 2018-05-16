from __future__ import absolute_import
import sklearn.neighbors
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN
from datasketch import MinHash


class LSHF(BaseANN):
    def __init__(self, metric, n_estimators=10, n_candidates=50):
        self.name = 'LSHF(n_est=%d, n_cand=%d)' % (n_estimators, n_candidates)
        self._metric = metric
        self._n_estimators = n_estimators
        self._n_candidates = n_candidates

    def fit(self, X):
        self.index = numpy.empty([0, 128])
        for i, x in enumerate(X):
            m = MinHash(num_perm=128)
            for e in x:
                m.update(str(e).encode('utf-8'))
            self.index = numpy.vstack((self.index, m.digest()))
            self._index_minhash.append(m)
        self._lshf = sklearn.neighbors.LSHForest(
            n_estimators=self._n_estimators, n_candidates=self._n_candidates)
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._lshf.fit(self.index)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize([v], axis=1, norm='l2')[0]
        m = MinHash(num_perm=128)
        for e in v:
            m.update(str(e).encode('utf-8'))
        return self._lshf.kneighbors(
            [m.digest()], return_distance=False, n_neighbors=n)[0]
