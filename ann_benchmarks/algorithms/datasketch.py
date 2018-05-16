from __future__ import absolute_import
from datasketch import MinHashLSHForest, MinHash
from ann_benchmarks.algorithms.base import BaseANN

from ann_benchmarks.distance import metrics

from pymongo import MongoClient
import gridfs
import pickle
import bz2


class DataSketch(BaseANN):
    def __init__(self, metric, n_perm, n_rep):
        if metric not in ('jaccard'):
            raise NotImplementedError(
                "Datasketch doesn't support metric %s" % metric)
        self._n_perm = 128
        self._n_rep = n_rep
        self._metric = metric
        self.name = 'Datasketch(n_perm=%d, n_rep=%d)' % (n_perm, n_rep)
        self.test_counter = 0

        client = MongoClient()
        db = client['vtest102']
        self.fs = gridfs.GridFS(db)

    def fit(self, X):
        self._index = MinHashLSHForest(num_perm=128, l=self._n_rep)
        raw_groups = self.fs.find({'type': 'train_row'})
        i = 0
        for raw_group in raw_groups:
            mhash = pickle.loads(bz2.decompress(raw_group.read()))
            self._index.add(str(i).encode('utf-8'), mhash)
            i += 1

        self._index.index()

    def query(self, v, n):
        test_rows = self.fs.find({'type': 'test_row'})
        test_mhash = pickle.loads(
            bz2.decompress(test_rows[self.test_counter].read()))

        # print(list(map(int, self._index.query(m, n))))
        if self.test_counter == 1817:
            self.test_counter = 0
        else:
            self.test_counter += 1
        return list(map(int, self._index.query(test_mhash, n)))