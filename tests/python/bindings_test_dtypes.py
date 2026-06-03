import json
import os
import pickle
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import hnswlib

sys.path.append(str(Path(__file__).resolve().parents[2] / "examples" / "python"))
from convert_bigann_index_to_u8 import convert_index


def _tmp_index_path():
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    os.remove(path)
    return path


def _bruteforce_l2(data, queries, k):
    diffs = queries[:, None, :].astype(np.float32) - data[None, :, :].astype(np.float32)
    distances = np.sum(diffs * diffs, axis=2)
    return np.argsort(distances, axis=1)[:, :k], np.sort(distances, axis=1)[:, :k]


class TypedIndexTestCase(unittest.TestCase):
    def test_default_float32_stays_legacy(self):
        dim = 8
        data = np.random.RandomState(17).random_sample((20, dim)).astype(np.float32)
        path = _tmp_index_path()
        try:
            index = hnswlib.Index(space="l2", dim=dim)
            self.assertEqual(index.dtype, "float32")
            index.init_index(max_elements=len(data), M=8, ef_construction=40)
            index.add_items(data, np.arange(len(data)))
            index.save_index(path)

            self.assertFalse(os.path.exists(path + ".hnswmeta.json"))

            loaded = hnswlib.Index(space="l2", dim=dim)
            loaded.load_index(path)
            labels, _ = loaded.knn_query(data[:3], k=1)
            np.testing.assert_array_equal(labels.ravel(), np.arange(3))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_uint8_persistence_metadata_and_exact_search(self):
        self._exercise_integer_dtype("uint8")

    def test_int8_persistence_metadata_and_exact_search(self):
        self._exercise_integer_dtype("int8")

    def _exercise_integer_dtype(self, dtype):
        dim = 16
        num_elements = 48
        rng = np.random.RandomState(11)
        if dtype == "uint8":
            data = rng.randint(0, 256, size=(num_elements, dim), dtype=np.uint8)
            replacement = np.full((1, dim), 255, dtype=np.uint8)
        else:
            data = rng.randint(-128, 128, size=(num_elements, dim)).astype(np.int8)
            replacement = np.full((1, dim), -128, dtype=np.int8)

        path = _tmp_index_path()
        try:
            index = hnswlib.Index(space="l2", dim=dim, dtype=dtype)
            self.assertEqual(index.dtype, dtype)
            index.init_index(max_elements=num_elements + 4, M=16, ef_construction=80, allow_replace_deleted=True)
            index.set_ef(64)
            index.add_items(data, np.arange(num_elements))

            returned = index.get_items([0, 1, 2])
            self.assertEqual(returned.dtype, data.dtype)
            np.testing.assert_array_equal(returned, data[:3])

            expected_labels, expected_distances = _bruteforce_l2(data, data[:5], k=3)
            labels, distances = index.knn_query(data[:5], k=3)
            np.testing.assert_array_equal(labels, expected_labels)
            np.testing.assert_allclose(distances, expected_distances)

            with self.assertRaises(Exception):
                index.add_items(data.astype(np.float32), [num_elements])
            with self.assertRaises(Exception):
                index.knn_query(data[:1].astype(np.float32), k=1)

            index.mark_deleted(0)
            index.add_items(replacement, [1000], replace_deleted=True)
            labels, _ = index.knn_query(replacement, k=1)
            self.assertEqual(labels[0, 0], 1000)
            index.resize_index(num_elements + 8)
            self.assertEqual(index.max_elements, num_elements + 8)

            index.save_index(path)
            with open(path + ".hnswmeta.json") as meta_file:
                metadata = json.load(meta_file)
            self.assertEqual(metadata["format"], "hnswlib-typed-index-metadata")
            self.assertEqual(metadata["version"], 1)
            self.assertEqual(metadata["space"], "l2")
            self.assertEqual(metadata["dtype"], dtype)
            self.assertEqual(metadata["dim"], dim)
            self.assertEqual(metadata["distance_type"], "float32")
            self.assertEqual(metadata["payload_bytes_per_vector"], dim)
            self.assertEqual(metadata["label_offset"] - metadata["offset_data"], dim)
            self.assertEqual(metadata["index_file_size"], os.path.getsize(path))

            loaded = hnswlib.Index(space="l2", dim=dim, dtype=dtype)
            loaded.load_index(path, allow_replace_deleted=True)
            self.assertEqual(loaded.dtype, dtype)
            self.assertEqual(loaded.get_items([1]).dtype, data.dtype)
            labels, _ = loaded.knn_query(replacement, k=1)
            self.assertEqual(labels[0, 0], 1000)

            unpickled = pickle.loads(pickle.dumps(loaded))
            self.assertEqual(unpickled.dtype, dtype)
            self.assertEqual(unpickled.get_items([1]).dtype, data.dtype)

            wrong_dtype = "int8" if dtype == "uint8" else "uint8"
            with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
                hnswlib.Index(space="l2", dim=dim, dtype=wrong_dtype).load_index(path)

            os.remove(path + ".hnswmeta.json")
            with self.assertRaisesRegex(RuntimeError, "require sidecar metadata"):
                hnswlib.Index(space="l2", dim=dim, dtype=dtype).load_index(path)
        finally:
            for suffix in ("", ".hnswmeta.json"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)

    def test_integer_dtypes_are_l2_only(self):
        with self.assertRaisesRegex(RuntimeError, "only for space='l2'"):
            hnswlib.Index(space="ip", dim=4, dtype="uint8")
        with self.assertRaisesRegex(RuntimeError, "only for space='l2'"):
            hnswlib.Index(space="cosine", dim=4, dtype="int8")

    def test_convert_float_hnsw_payload_to_uint8_index(self):
        self._exercise_float_hnsw_to_uint8_conversion("base.u8bin")

    def test_convert_float_hnsw_payload_with_fbin_base_to_uint8_index(self):
        self._exercise_float_hnsw_to_uint8_conversion("base.fbin")

    def _exercise_float_hnsw_to_uint8_conversion(self, base_filename):
        dim = 12
        num_elements = 40
        rng = np.random.RandomState(23)
        data_u8 = rng.randint(0, 256, size=(num_elements, dim), dtype=np.uint8)
        queries_u8 = data_u8[:6]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, base_filename)
            float_index_path = os.path.join(tmpdir, "source_float.bin")
            converted_path = os.path.join(tmpdir, "converted_uint8.bin")

            with open(base_path, "wb") as output:
                output.write(struct.pack("<ii", num_elements, dim))
                if base_filename.endswith(".fbin"):
                    output.write(data_u8.astype(np.float32).tobytes())
                else:
                    output.write(data_u8.tobytes())

            float_index = hnswlib.Index(space="l2", dim=dim)
            float_index.init_index(max_elements=num_elements, M=8, ef_construction=60)
            float_index.set_ef(64)
            float_index.add_items(data_u8.astype(np.float32), np.arange(num_elements))
            expected_labels, expected_distances = float_index.knn_query(queries_u8.astype(np.float32), k=5)
            float_index.save_index(float_index_path)

            result = convert_index(float_index_path, base_path, converted_path, verify_samples=num_elements)
            self.assertEqual(result["mode"], "converted_float32_payload")
            self.assertEqual(result["base_dtype"], "float32" if base_filename.endswith(".fbin") else "uint8")
            self.assertEqual(result["source_payload_bytes"], dim * 4)
            self.assertEqual(result["output_payload_bytes"], dim)
            self.assertEqual(result["rows_verified_during_conversion"], num_elements)
            self.assertTrue(os.path.exists(converted_path + ".hnswmeta.json"))

            converted = hnswlib.Index(space="l2", dim=dim, dtype="uint8")
            converted.load_index(converted_path)
            converted.set_ef(64)
            np.testing.assert_array_equal(converted.get_items([0, 7, 13]), data_u8[[0, 7, 13]])

            labels, distances = converted.knn_query(queries_u8, k=5)
            np.testing.assert_array_equal(labels, expected_labels)
            np.testing.assert_allclose(distances, expected_distances)


if __name__ == "__main__":
    unittest.main()
