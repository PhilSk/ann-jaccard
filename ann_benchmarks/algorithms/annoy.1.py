from __future__ import absolute_import
import annoy
from ann_benchmarks.algorithms.base import BaseANN
import numpy
from datasketch import MinHash


class Annoy(BaseANN):
    def __init__(self, metric, n_trees, search_k):
        self._n_trees = n_trees
        self._search_k = search_k
        self._metric = metric
        self.name = 'Annoy(n_trees=%d, search_k=%d)' % (self._n_trees,
                                                        self._search_k)

    def fit(self, X):
        self.index = numpy.empty([0, 1])

        self._index_minhash = []
        for i, x in enumerate(X):
            m = MinHash(num_perm=1)
            for e in x:
                m.update(str(e).encode('utf-8'))
            self.index = numpy.vstack((self.index, m.digest()))
            self._index_minhash.append(m)
        self._annoy = annoy.AnnoyIndex(self.index.shape[1])
        for i, x in enumerate(self.index):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def query(self, v, n):
        m = MinHash(num_perm=1)
        for e in v:
            m.update(str(e).encode('utf-8'))
        print(self._annoy.get_nns_by_vector(m.digest().tolist(), n,
                                            self._search_k))
        return self._annoy.get_nns_by_vector(m.digest().tolist(), n,
                                             self._search_k)
