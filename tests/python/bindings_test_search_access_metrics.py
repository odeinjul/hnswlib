import unittest
import tempfile
from pathlib import Path

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

    def test_readonly_mmap_matches_regular_load(self):
        dimension = 8
        point_count = 512
        rng = np.random.RandomState(29)
        data = rng.random_sample((point_count, dimension)).astype(np.float32)
        queries = rng.random_sample((16, dimension)).astype(np.float32)

        built = hnswlib.Index(space="l2", dim=dimension)
        built.init_index(
            max_elements=point_count,
            M=16,
            ef_construction=100,
            random_seed=100,
        )
        built.add_items(data, np.arange(point_count))
        built.set_ef(64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.bin"
            built.save_index(str(path))

            regular = hnswlib.Index(space="l2", dim=dimension)
            regular.load_index(str(path))
            regular.set_ef(64)
            regular_labels, regular_distances = regular.knn_query(
                queries, k=10, num_threads=1
            )

            mapped = hnswlib.Index(space="l2", dim=dimension)
            mapped.load_index_readonly_mmap(str(path))
            mapped.set_ef(64)
            mapped.set_search_access_metrics_enabled(True)
            mapped_labels, mapped_distances = mapped.knn_query(
                queries, k=10, num_threads=1
            )

        np.testing.assert_array_equal(mapped_labels, regular_labels)
        np.testing.assert_allclose(mapped_distances, regular_distances)
        metrics = mapped.get_search_access_metrics()
        self.assertEqual(metrics["entrypoint_vector_accesses"], len(queries))
        self.assertGreater(metrics["l0_neighbor_list_accesses"], 0)


if __name__ == "__main__":
    unittest.main()
