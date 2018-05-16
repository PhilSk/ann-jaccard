from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
from .hyperminhash_class import HyperMinHash
import numpy as np


class BloomFilter(BaseANN):
    def __init__(self, metric, bucketbits, bucketsize, subbucketsize):
        print(metric)
        if metric not in ['jaccard']:
            raise NotImplementedError(
                "HyperMinHash doesn't support metric %s" % metric)
        self._bucketbits = bucketbits
        self._bucketsize = bucketsize
        self._subbucketsize = subbucketsize
        self._metric = metric
        self.name = 'Hyperminhash(bucketbits=%d, bucketsize=%d)' % (
            bucketbits, bucketsize)

    def fit(self, X):
        self._source = []
        self._storage = []
        self._result = []
        for i, x in enumerate(X):
            self._source.append(x)
            hmh = HyperMinHash(self._bucketbits,
                               self._bucketsize, self._subbucketsize)
            hmh.update(x)
            self._storage.append(hmh)
        print("N")

    def query(self, v, n):
        hmh_query = HyperMinHash(self._bucketbits,
                                 self._bucketsize, self._subbucketsize)
        hmh_query.update(v)

        for i in range(len(self._source)):
            self._result.append(self._storage[i].intersection(hmh_query)[1])
        arr = np.array(self._result)
        res = arr.argsort()[-n:][::-1]
        self._result = []
        print(v)
        print(res)
        return res
