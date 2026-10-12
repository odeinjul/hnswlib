import unittest

import numpy as np

import hnswlib


class InsertConcurrencyTestCase(unittest.TestCase):
    def testModeIsLockedByDefaultAndValidated(self):
        index = hnswlib.Index(space='l2', dim=4)
        with self.assertRaises(RuntimeError):
            index.set_insert_concurrency('relaxed')
        index.init_index(max_elements=10, ef_construction=20, M=4)
        self.assertEqual(index.get_insert_concurrency(), 'locked')
        index.set_insert_concurrency('relaxed')
        self.assertEqual(index.get_insert_concurrency(), 'relaxed')
        with self.assertRaises(ValueError):
            index.set_insert_concurrency('unlocked')

    def testRelaxedConcurrentInsertsKeepTheGraphSearchable(self):
        # Small M and ef_construction with many threads make inserts race on the same lists;
        # repeats give the race several chances to surface as an exception.
        rng = np.random.default_rng(7)
        for _ in range(3):
            data = rng.random((20000, 8), dtype=np.float32)
            index = hnswlib.Index(space='l2', dim=8)
            index.init_index(max_elements=len(data), ef_construction=32, M=4)
            index.add_items(data[:1000], np.arange(1000), num_threads=1)
            index.set_insert_concurrency('relaxed')
            index.add_items(data[1000:], np.arange(1000, len(data)), num_threads=8)

            self.assertEqual(index.get_current_count(), len(data))
            self.assertEqual(sorted(index.get_ids_list()), list(range(len(data))))
            index.set_ef(100)
            labels, _ = index.knn_query(data[:500], k=1, num_threads=4)
            self.assertGreater(np.mean(labels[:, 0] == np.arange(500)), 0.9)

    def testLinkOrderIsPerLevelByDefaultAndValidated(self):
        index = hnswlib.Index(space='l2', dim=4)
        with self.assertRaises(RuntimeError):
            index.set_insert_link_order('own-lists-first')
        index.init_index(max_elements=10, ef_construction=20, M=4)
        self.assertEqual(index.get_insert_link_order(), 'per-level')
        index.set_insert_link_order('own-lists-first')
        self.assertEqual(index.get_insert_link_order(), 'own-lists-first')
        with self.assertRaises(ValueError):
            index.set_insert_link_order('lists-first')

    def testOwnListsFirstInsertsBuildTheSameQualityGraph(self):
        rng = np.random.default_rng(11)
        data = rng.random((20000, 8), dtype=np.float32)
        recalls = {}
        for concurrency in ('locked', 'relaxed'):
            for order in ('per-level', 'own-lists-first'):
                index = hnswlib.Index(space='l2', dim=8)
                index.init_index(max_elements=len(data), ef_construction=64, M=8, random_seed=3)
                index.set_insert_concurrency(concurrency)
                index.set_insert_link_order(order)
                index.add_items(data, np.arange(len(data)), num_threads=8)
                self.assertEqual(sorted(index.get_ids_list()), list(range(len(data))))
                index.set_ef(100)
                labels, _ = index.knn_query(data[:1000], k=1, num_threads=4)
                recalls[concurrency, order] = np.mean(labels[:, 0] == np.arange(1000))
        for (concurrency, order), recall in recalls.items():
            self.assertGreater(recall, 0.97, (concurrency, order))


if __name__ == '__main__':
    unittest.main()
