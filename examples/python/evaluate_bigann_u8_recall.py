import argparse
import struct
import time

import numpy as np

import hnswlib


def _float_rows_to_uint8(rows):
    if not np.all(np.isfinite(rows)):
        raise RuntimeError("fbin query vectors contain non-finite values")
    rounded = np.rint(rows)
    if not np.array_equal(rows, rounded):
        raise RuntimeError("fbin query vectors contain non-integer values and cannot be converted exactly to uint8")
    if np.any(rounded < 0) or np.any(rounded > 255):
        raise RuntimeError("fbin query vectors contain values outside uint8 range")
    return rounded.astype(np.uint8)


def read_vector_file(path, max_vectors=None):
    if path.endswith(".u8bin"):
        dtype = np.uint8
    elif path.endswith(".fbin"):
        dtype = np.float32
    else:
        raise RuntimeError("query file must be a BigANN-style .u8bin or .fbin file")

    with open(path, "rb") as input_file:
        header = np.fromfile(input_file, dtype=np.int32, count=2)
        if header.size != 2:
            raise RuntimeError("query file is missing the two-int32 header")
        num_vectors, dim = int(header[0]), int(header[1])
        if max_vectors is not None:
            num_vectors = min(num_vectors, max_vectors)
        data = np.fromfile(input_file, dtype=dtype, count=num_vectors * dim)
    if data.size != num_vectors * dim:
        raise RuntimeError("query file ended before the requested vectors were read")
    data = data.reshape(num_vectors, dim)
    if dtype == np.float32:
        data = _float_rows_to_uint8(data)
    return data


def read_groundtruth_ids(path, query_count, k):
    with open(path, "rb") as input_file:
        n, stored_k = struct.unpack("<II", input_file.read(8))
        if query_count > n:
            raise RuntimeError("groundtruth has fewer query rows than requested")
        if k > stored_k:
            raise RuntimeError("groundtruth has fewer neighbors than requested")
        ids = np.fromfile(input_file, dtype=np.uint32, count=n * stored_k).reshape(n, stored_k)
    return ids[:query_count, :k]


def recall_at_k(labels, groundtruth):
    hits = 0
    for found, expected in zip(labels, groundtruth):
        hits += len(set(int(x) for x in found) & set(int(x) for x in expected))
    return hits / float(labels.shape[0] * labels.shape[1])


def main():
    parser = argparse.ArgumentParser(description="Evaluate recall for a native uint8 BigANN HNSW index.")
    parser.add_argument("index")
    parser.add_argument("query_vectors")
    parser.add_argument("groundtruth_bin")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=100)
    parser.add_argument("--threads", type=int, default=-1)
    args = parser.parse_args()

    queries = read_vector_file(args.query_vectors, args.queries)
    if queries.shape[1] != args.dim:
        raise RuntimeError("query dimensionality does not match --dim")
    groundtruth = read_groundtruth_ids(args.groundtruth_bin, queries.shape[0], args.k)

    index = hnswlib.Index(space="l2", dim=args.dim, dtype="uint8")
    index.load_index(args.index)
    index.set_ef(args.ef)
    index.set_num_threads(args.threads)

    start = time.time()
    labels, distances = index.knn_query(queries, k=args.k, num_threads=args.threads)
    elapsed = time.time() - start

    print("queries:", queries.shape[0])
    print("k:", args.k)
    print("ef:", args.ef)
    print("threads:", args.threads)
    print("recall_at_%d:" % args.k, "%.6f" % recall_at_k(labels, groundtruth))
    print("query_seconds:", "%.3f" % elapsed)
    print("qps:", "%.3f" % (queries.shape[0] / elapsed))
    print("distance_dtype:", distances.dtype)


if __name__ == "__main__":
    main()
