import unittest

import numpy as np

import hnswlib


class SearchAccessMetricsTestCase(unittest.TestCase):
    def test_metrics_are_opt_in_resettable_and_available_from_python(self):
        dimension = 8
        point_count = 512
        query_count = 16
        rng = np.random.RandomState(23)
        data = rng.random_sample((point_count, dimension)).astype(np.float32)
        queries = rng.random_sample((query_count, dimension)).astype(np.float32)

        index = hnswlib.Index(space="l2", dim=dimension)
        index.init_index(
            max_elements=point_count,
            M=16,
            ef_construction=100,
            random_seed=100,
        )
        index.add_items(data, np.arange(point_count))
        index.set_ef(64)

        initial = index.get_search_access_metrics()
        self.assertFalse(initial["enabled"])
        index.knn_query(queries, k=10, num_threads=4)
        disabled = index.get_search_access_metrics()
        self.assertEqual(disabled["vector_accesses"], 0)
        self.assertEqual(disabled["neighbor_list_accesses"], 0)

        index.set_search_access_metrics_enabled(True)
        index.reset_search_access_metrics()
        index.knn_query(queries, k=10, num_threads=4)
        metrics = index.get_search_access_metrics()

        self.assertTrue(metrics["enabled"])
        self.assertEqual(
            metrics["entrypoint_vector_accesses"],
            query_count,
        )
        self.assertGreater(metrics["upper_neighbor_list_accesses"], 0)
        self.assertGreater(metrics["upper_vector_accesses"], 0)
        self.assertGreater(metrics["l0_neighbor_list_accesses"], 0)
        self.assertGreater(metrics["l0_vector_accesses"], 0)
        self.assertEqual(
            metrics["neighbor_list_accesses"],
            metrics["upper_neighbor_list_accesses"]
            + metrics["l0_neighbor_list_accesses"],
        )
        self.assertEqual(
            metrics["vector_accesses"],
            metrics["entrypoint_vector_accesses"]
            + metrics["upper_vector_accesses"]
            + metrics["l0_vector_accesses"],
        )

        index.reset_search_access_metrics()
        reset = index.get_search_access_metrics()
        self.assertTrue(reset["enabled"])
        self.assertEqual(reset["vector_accesses"], 0)
        self.assertEqual(reset["neighbor_list_accesses"], 0)

        index.set_search_access_metrics_enabled(False)
        self.assertFalse(index.get_search_access_metrics()["enabled"])


if __name__ == "__main__":
    unittest.main()
