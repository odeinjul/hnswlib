import argparse
import os
import time

import numpy as np

import hnswlib

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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


def batch_ranges(total, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def add_items_with_progress(index, data, labels, num_threads, batch_size, show_progress):
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=data.shape[0], unit="vectors", desc="Building HNSW index")

    try:
        for start, end in batch_ranges(data.shape[0], batch_size):
            index.add_items(data[start:end], labels[start:end], num_threads=num_threads)
            if progress is not None:
                progress.update(end - start)
    finally:
        if progress is not None:
            progress.close()


def main():
    parser = argparse.ArgumentParser(description="Build a native uint8 HNSW index from a BigANN .u8bin file.")
    parser.add_argument("input", help="Path to a BigANN-style .u8bin file")
    parser.add_argument("output", help="Path for the HNSW index")
    parser.add_argument("--max-vectors", type=int, default=100000, help="Number of vectors to read")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef", type=int, default=100)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=100000, help="Rows per add_items call")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar while adding vectors",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")

    data = read_u8bin(args.input, args.max_vectors)
    labels = np.arange(data.shape[0], dtype=np.uint64)

    index = hnswlib.Index(space="l2", dim=data.shape[1], dtype="uint8")
    index.init_index(max_elements=data.shape[0], M=args.M, ef_construction=args.ef_construction)
    index.set_ef(args.ef)

    start = time.time()
    add_items_with_progress(
        index,
        data,
        labels,
        num_threads=args.threads,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
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
