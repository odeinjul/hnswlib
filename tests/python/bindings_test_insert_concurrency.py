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

if __name__ == '__main__':
    unittest.main()
