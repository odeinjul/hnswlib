import argparse
import os
import time

import numpy as np

import hnswlib


def read_u8bin(path, max_vectors=None):
    with open(path, "rb") as input_file:
        header = np.fromfile(input_file, dtype=np.int32, count=2)
        if header.size != 2:
            raise RuntimeError("u8bin file is missing the two-int32 header")
        num_vectors, dim = int(header[0]), int(header[1])
        if num_vectors < 0 or dim <= 0:
            raise RuntimeError("u8bin header has invalid shape")

        if max_vectors is not None:
            num_vectors = min(num_vectors, max_vectors)

        data = np.fromfile(input_file, dtype=np.uint8, count=num_vectors * dim)
        if data.size != num_vectors * dim:
            raise RuntimeError("u8bin file ended before the requested vectors were read")
        return data.reshape(num_vectors, dim)


def main():
    parser = argparse.ArgumentParser(description="Build a native uint8 HNSW index from a BigANN .u8bin file.")
    parser.add_argument("input", help="Path to a BigANN-style .u8bin file")
    parser.add_argument("output", help="Path for the HNSW index")
    parser.add_argument("--max-vectors", type=int, default=100000, help="Number of vectors to read")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef", type=int, default=100)
    parser.add_argument("--threads", type=int, default=-1)
    args = parser.parse_args()

    data = read_u8bin(args.input, args.max_vectors)
    labels = np.arange(data.shape[0], dtype=np.uint64)

    index = hnswlib.Index(space="l2", dim=data.shape[1], dtype="uint8")
    index.init_index(max_elements=data.shape[0], M=args.M, ef_construction=args.ef_construction)
    index.set_ef(args.ef)

    start = time.time()
    index.add_items(data, labels, num_threads=args.threads)
    build_seconds = time.time() - start

    index.save_index(args.output)
    index_size = os.path.getsize(args.output)
    meta_size = os.path.getsize(args.output + ".hnswmeta.json")

    print("vectors:", data.shape[0])
    print("dim:", data.shape[1])
    print("build_seconds:", "%.3f" % build_seconds)
    print("index_file_size:", index_size)
    print("metadata_file_size:", meta_size)


if __name__ == "__main__":
    main()
