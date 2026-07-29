#!/usr/bin/env python3
"""Measure logical HNSW search accesses for one query-only run."""

from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import hnswlib
import numpy as np


def read_fbin(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"truncated fbin header: {path}")
    rows, dimensions = (int(value) for value in header)
    expected_bytes = 8 + rows * dimensions * np.dtype(np.float32).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"invalid fbin size for {path}: expected {expected_bytes}, "
            f"got {path.stat().st_size}"
        )
    return np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(rows, dimensions),
    )


def read_groundtruth(path: Path, query_count: int, k: int) -> np.ndarray:
    header = np.fromfile(path, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"truncated ground-truth header: {path}")
    rows, width = (int(value) for value in header)
    if rows < query_count or width < k:
        raise ValueError(
            f"ground truth {path} has shape ({rows}, {width}), "
            f"need at least ({query_count}, {k})"
        )
    ids_bytes = rows * width * np.dtype(np.uint32).itemsize
    expected_bytes = 8 + 2 * ids_bytes
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"invalid ground-truth size for {path}: expected "
            f"{expected_bytes}, got {path.stat().st_size}"
        )
    ids = np.memmap(
        path,
        dtype=np.uint32,
        mode="r",
        offset=8,
        shape=(rows, width),
    )
    return np.asarray(ids[:query_count, :k])


def recall_at_k(labels: np.ndarray, groundtruth: np.ndarray) -> float:
    matches = sum(
        len(set(found).intersection(expected))
        for found, expected in zip(labels, groundtruth)
    )
    return matches / groundtruth.size


def git_state(repo: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"branch": branch, "commit": commit, "dirty": bool(status)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--space", default="l2", choices=("l2", "ip", "cosine"))
    parser.add_argument("--ef-search", type=int, default=150)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="number of queries to run; 0 uses the complete query file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ef_search < args.k:
        raise ValueError("--ef-search must be at least --k")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    queries = read_fbin(args.queries)
    query_count = len(queries)
    if args.query_limit:
        if args.query_limit < 0 or args.query_limit > query_count:
            raise ValueError("--query-limit is outside the query file")
        query_count = args.query_limit
    queries = np.asarray(queries[:query_count])
    groundtruth = read_groundtruth(args.groundtruth, query_count, args.k)

    started_at = datetime.now(timezone.utc)
    load_started = time.perf_counter()
    index = hnswlib.Index(
        space=args.space,
        dim=queries.shape[1],
        dtype="float32",
    )
    index.load_index_readonly_mmap(str(args.index))
    load_seconds = time.perf_counter() - load_started
    if index.get_current_count() != 100_000_000:
        raise RuntimeError(
            f"expected a 100M index, got {index.get_current_count()} elements"
        )

    index.set_ef(args.ef_search)
    index.set_search_access_metrics_enabled(True)
    index.reset_search_access_metrics()
    search_started = time.perf_counter()
    labels, _ = index.knn_query(
        queries,
        k=args.k,
        num_threads=args.threads,
    )
    search_seconds = time.perf_counter() - search_started
    metrics = dict(index.get_search_access_metrics())
    if metrics["entrypoint_vector_accesses"] != query_count:
        raise RuntimeError(
            "entry-point count does not match completed query count: "
            f"{metrics['entrypoint_vector_accesses']} != {query_count}"
        )

    totals = {
        key: int(metrics[key])
        for key in (
            "entrypoint_vector_accesses",
            "upper_neighbor_list_accesses",
            "upper_vector_accesses",
            "l0_neighbor_list_accesses",
            "l0_vector_accesses",
            "neighbor_list_accesses",
            "vector_accesses",
        )
    }
    per_query = {key: value / query_count for key, value in totals.items()}
    repo = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": 1,
        "experiment": "M2",
        "measurement": "logical_hnsw_search_accesses",
        "dataset": "DEEP-100M",
        "workload": "query-only",
        "run_count": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.getfqdn(),
        "pid": os.getpid(),
        "source": git_state(repo),
        "inputs": {
            "index": str(args.index.resolve()),
            "index_bytes": args.index.stat().st_size,
            "queries": str(args.queries.resolve()),
            "groundtruth": str(args.groundtruth.resolve()),
            "index_mode": "read-only-mmap",
            "pq": False,
            "explicit_cache": False,
        },
        "parameters": {
            "space": args.space,
            "dimension": int(queries.shape[1]),
            "index_elements": index.get_current_count(),
            "query_count": query_count,
            "k": args.k,
            "ef_search": args.ef_search,
            "threads": args.threads,
        },
        "results": {
            "recall_at_10": recall_at_k(labels, groundtruth),
            "load_seconds": load_seconds,
            "search_seconds": search_seconds,
            "throughput_qps": query_count / search_seconds,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "access_totals": totals,
            "accesses_per_query": per_query,
        },
        "semantics": {
            "entrypoint_vector_accesses": (
                "initial distance evaluation against the entry-point vector"
            ),
            "neighbor_list_access": (
                "one logical visit to one HNSW adjacency list; list header and "
                "neighbor IDs count together"
            ),
            "vector_access": (
                "one vector used as an input to the query distance function"
            ),
            "excluded": (
                "speculative CPU prefetches, visited-set accesses, queue state, "
                "PQ, caches, and physical RDMA operations"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
