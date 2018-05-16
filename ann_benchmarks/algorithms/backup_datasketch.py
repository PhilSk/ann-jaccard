from __future__ import absolute_import
from datasketch import MinHashLSHForest, MinHash
from ann_benchmarks.algorithms.base import BaseANN
from sklearn.neighbors import KDTree, BallTree
import sklearn.neighbors

import numpy
from ann_benchmarks.distance import metrics as pd

import annoy


class DataSketch(BaseANN):
    def __init__(self, metric, n_perm, n_rep, n_leaves):
        if metric not in ('jaccard'):
            raise NotImplementedError(
                "Datasketch doesn't support metric %s" % metric)
        self._n_perm = n_perm
        self._n_rep = n_rep
        self._n_leaves = n_leaves
        self._metric = metric
        self.name = 'Datasketch(n_perm=%d, n_rep=%d, n_leaves=%d)' % (n_perm,
                                                                      n_rep,
                                                                      n_leaves)

    def fit(self, X):
        self.index = numpy.empty([0, 32])
        self._index_minhash = []
        self._ball_index = []
        self._index = MinHashLSHForest(num_perm=self._n_perm, l=self._n_rep)

        for i, x in enumerate(X):
            m = MinHash(num_perm=self._n_perm)
            for e in x:
                m.update(str(e).encode('utf-8'))
            self._index.add(str(i), m)
            #self.index.append(m.digest())
            self.index = numpy.vstack((self.index, m.digest()))
            self._ball_index.append(m.digest())
            self._index_minhash.append(m)
        self._index.index()
        self._X = X

        self.tree = BallTree(self.index, leaf_size=self._n_leaves)

        # self._annoy = annoy.AnnoyIndex(X.shape[1], metric='euclidean')
        # for i, x in enumerate(X):
        #     self._annoy.add_item(i, x.tolist())
        # self._annoy.build(100)

    def query(self, v, n):
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        m = MinHash(num_perm=self._n_perm)
        for e in v:
            m.update(str(e).encode('utf-8'))

        # for i in self._annoy.get_nns_by_vector(v.tolist(), n, 100):
        #     print(self._index_minhash[int(i)].jaccard(m))

        dist, ind = self.tree.query([m.digest()], k=n)
        for i in ind[0]:
            # print(i)
            print(self._index_minhash[int(i)].jaccard(m))
        print("=======================")
        brute_indices = self.query_with_distances(m.digest(), n)
        for i in brute_indices:
            print(self._index_minhash[int(i)].jaccard(m))
        print("-----------------------")
        ind2 = self._index.query(m, n)
        for i in ind2:
            print(self._index_minhash[int(i)].jaccard(m))

        # return map(int, ind[0])
        return self.query_with_distances(m.digest(), n)

    popcount = []
    for i in range(256):
        popcount.append(bin(i).count("1"))

    def query_with_distances(self, v, n):
        """Find indices of `n` most similar vectors from the index to query vector `v`."""
        if self._metric == 'jaccard':
            dists = numpy.array(
                [pd[self._metric]['distance'](v, e) for e in self.index])
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!
        # partition-sort by distance, get `n` closest
        nearest_indices = dists.argsort()[-n:][::-1]

        return nearest_indices