#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


UINT32_MAX = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True)
class U8BinHeader:
    count: int
    dim: int


@dataclass(frozen=True)
class GtHeader:
    num_queries: int
    topk: int


@dataclass(frozen=True)
class BatchSpec:
    batch_index: int
    active_begin: int
    active_end_exclusive: int
    active_segments: tuple[tuple[int, int], ...]
    insert_begin: int
    insert_end_exclusive: int
    delete_begin: int
    delete_end_exclusive: int
    insert_ids: np.ndarray
    delete_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a uint8 SIFT10M/BigANN10M rolling update-eval workload."
    )
    parser.add_argument(
        "--base-u8bin-path",
        default="/data/vectordb-cxl/data/bigann-10m-uint8/base.u8bin",
    )
    parser.add_argument(
        "--query-u8bin-path",
        default="/data/vectordb-cxl/data/bigann-10m-uint8/unif_query_10k.u8bin",
    )
    parser.add_argument(
        "--groundtruth-path",
        default="/data/vectordb-cxl/data/bigann-10m-uint8/unif_groundtruth_10k.bin",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/vectordb-cxl/data/sift10m-update-eval-uint8",
    )
    parser.add_argument("--total-vectors", type=int, default=10_000_000)
    parser.add_argument("--initial-active", type=int, default=8_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument(
        "--base-link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How base.u8bin and query artifacts are placed in the output tree.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.total_vectors <= 0:
        parser.error("--total-vectors must be > 0")
    if args.initial_active <= 0:
        parser.error("--initial-active must be > 0")
    if args.initial_active > args.total_vectors:
        parser.error("--initial-active cannot exceed --total-vectors")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.total_vectors % args.batch_size != 0:
        parser.error("--total-vectors must be divisible by --batch-size")
    if args.initial_active % args.batch_size != 0:
        parser.error("--initial-active must be divisible by --batch-size")
    if args.num_batches <= 0:
        parser.error("--num-batches must be > 0")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be > 0")
    return args


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def place_file(src: Path, dst: Path, mode: str, force: bool) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(f"{dst} exists; pass --force to overwrite")
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    raise ValueError(f"unsupported link mode: {mode}")


def read_u8bin_header(path: Path) -> U8BinHeader:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"invalid u8bin header in {path}")
    count, dim = struct.unpack("<II", raw)
    expected_size = 8 + count * dim
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"u8bin size mismatch for {path}: actual={actual_size}, expected={expected_size}"
        )
    return U8BinHeader(count=count, dim=dim)


def read_groundtruth(path: Path) -> tuple[GtHeader, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid groundtruth header in {path}")
        num_queries, topk = struct.unpack("<II", raw)
        ids = np.fromfile(f, dtype=np.uint32, count=num_queries * topk)
        if ids.size != num_queries * topk:
            raise ValueError(f"groundtruth ids truncated in {path}")
        dists = np.fromfile(f, dtype=np.float32, count=num_queries * topk)
        if dists.size != num_queries * topk:
            raise ValueError(f"groundtruth distances truncated in {path}")
    return GtHeader(num_queries, topk), ids.reshape(num_queries, topk), dists.reshape(num_queries, topk)


def write_groundtruth(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    if ids.dtype != np.uint32:
        raise ValueError("groundtruth ids must be uint32")
    if dists.dtype != np.float32:
        raise ValueError("groundtruth dists must be float32")
    if ids.shape != dists.shape:
        raise ValueError("groundtruth ids/dists shape mismatch")
    ensure_parent(path)
    with path.open("wb") as out:
        out.write(struct.pack("<II", ids.shape[0], ids.shape[1]))
        ids.tofile(out)
        dists.tofile(out)


def write_ids_bin(path: Path, ids: np.ndarray) -> None:
    if ids.dtype != np.uint32:
        raise ValueError("ids must be uint32")
    ensure_parent(path)
    ids.tofile(path)


def build_permutation(total_vectors: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_vectors, axis=0).astype(np.uint32, copy=False)
    if permutation.shape != (total_vectors,):
        raise ValueError("permutation shape invariant failed")
    if np.unique(permutation).size != total_vectors:
        raise ValueError("permutation is not bijective")
    return permutation


def circular_segments(begin: int, count: int, total_vectors: int) -> tuple[tuple[int, int], ...]:
    end = begin + count
    if end <= total_vectors:
        return ((begin, end),)
    return ((begin, total_vectors), (0, end % total_vectors))


def build_batch_spec(
    batch_idx: int,
    initial_active: int,
    batch_size: int,
    total_vectors: int,
) -> BatchSpec:
    total_chunks = total_vectors // batch_size
    initial_chunks = initial_active // batch_size
    if batch_idx == 0:
        active_begin = 0
        active_segments = ((0, initial_active),)
        insert_begin = insert_end = initial_active
        delete_begin = delete_end = 0
        insert_ids = np.empty(0, dtype=np.uint32)
        delete_ids = np.empty(0, dtype=np.uint32)
    else:
        active_begin = (batch_idx % total_chunks) * batch_size
        active_segments = circular_segments(active_begin, initial_active, total_vectors)
        insert_chunk = (initial_chunks + batch_idx - 1) % total_chunks
        delete_chunk = (batch_idx - 1) % total_chunks
        insert_begin = insert_chunk * batch_size
        insert_end = insert_begin + batch_size
        delete_begin = delete_chunk * batch_size
        delete_end = delete_begin + batch_size
        insert_ids = np.arange(insert_begin, insert_end, dtype=np.uint32)
        delete_ids = np.arange(delete_begin, delete_end, dtype=np.uint32)

    active_count = sum(end - begin for begin, end in active_segments)
    if active_count != initial_active:
        raise ValueError(f"batch {batch_idx} active count invariant failed")
    for begin, end in active_segments:
        if begin < 0 or begin >= end or end > total_vectors:
            raise ValueError(f"batch {batch_idx} active segment invariant failed")
    expected_update = 0 if batch_idx == 0 else batch_size
    if insert_ids.size != expected_update or delete_ids.size != expected_update:
        raise ValueError(f"batch {batch_idx} insert/delete size invariant failed")

    return BatchSpec(
        batch_index=batch_idx,
        active_begin=active_begin,
        active_end_exclusive=active_segments[-1][1],
        active_segments=active_segments,
        insert_begin=insert_begin,
        insert_end_exclusive=insert_end,
        delete_begin=delete_begin,
        delete_end_exclusive=delete_end,
        insert_ids=insert_ids,
        delete_ids=delete_ids,
    )


def active_segments_json(segments: tuple[tuple[int, int], ...]) -> list[dict]:
    return [{"begin": begin, "end_exclusive": end} for begin, end in segments]


def sha256_u32(ids: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(ids.view(np.uint8))
    return h.hexdigest()


def batch_json_object(batch: BatchSpec, active_count: int, gt_path: str, query_path: str) -> dict:
    return {
        "batch_index": batch.batch_index,
        "insert_ids": batch.insert_ids.tolist(),
        "delete_ids": batch.delete_ids.tolist(),
        "active_begin": batch.active_begin,
        "active_end_exclusive": batch.active_end_exclusive,
        "active_segments": active_segments_json(batch.active_segments),
        "active_count": active_count,
        "insert_sha256_u32": sha256_u32(batch.insert_ids),
        "delete_sha256_u32": sha256_u32(batch.delete_ids),
        "groundtruth_path": gt_path,
        "query_path": query_path,
    }


def batch_manifest_entry(batch: BatchSpec, active_count: int, gt_path: str, query_path: str, batch_path: str) -> dict:
    return {
        "batch_index": batch.batch_index,
        "active_begin": batch.active_begin,
        "active_end_exclusive": batch.active_end_exclusive,
        "active_segments": active_segments_json(batch.active_segments),
        "active_count": active_count,
        "insert_begin": batch.insert_begin,
        "insert_end_exclusive": batch.insert_end_exclusive,
        "delete_begin": batch.delete_begin,
        "delete_end_exclusive": batch.delete_end_exclusive,
        "insert_sha256_u32": sha256_u32(batch.insert_ids),
        "delete_sha256_u32": sha256_u32(batch.delete_ids),
        "groundtruth_path": gt_path,
        "query_path": query_path,
        "batch_json_path": batch_path,
    }


def materialize_permuted_u8bin(
    input_path: Path,
    output_path: Path,
    expected_count: int,
    dim: int,
    permutation: np.ndarray,
    chunk_rows: int,
    force: bool,
) -> None:
    if output_path.exists():
        if not force:
            raise FileExistsError(f"{output_path} exists; pass --force to overwrite")
        output_path.unlink()
    payload = np.memmap(input_path, dtype=np.uint8, mode="r", offset=8, shape=(expected_count, dim))
    ensure_parent(output_path)
    with output_path.open("wb") as out:
        out.write(struct.pack("<II", expected_count, dim))
        for start in tqdm(range(0, expected_count, chunk_rows), desc="Writing base_permuted.u8bin", unit="chunk"):
            end = min(start + chunk_rows, expected_count)
            rows = payload[permutation[start:end]]
            np.asarray(rows, dtype=np.uint8).tofile(out)


def remap_groundtruth_ids(gt_ids_old: np.ndarray, inverse_permutation: np.ndarray, total_vectors: int) -> np.ndarray:
    valid = gt_ids_old != UINT32_MAX
    if valid.any() and int(gt_ids_old[valid].max(initial=0)) >= total_vectors:
        raise ValueError("groundtruth contains id outside [0, total_vectors)")
    remapped = np.full(gt_ids_old.shape, UINT32_MAX, dtype=np.uint32)
    remapped[valid] = inverse_permutation[gt_ids_old[valid]]
    return remapped


def filter_groundtruth_for_active_segments(
    gt_ids_new: np.ndarray,
    gt_dists: np.ndarray,
    active_segments: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    num_queries, topk = gt_ids_new.shape
    keep_mask = np.zeros((num_queries, topk), dtype=bool)
    for begin, end in active_segments:
        keep_mask |= (gt_ids_new >= begin) & (gt_ids_new < end)
    kept_counts = keep_mask.sum(axis=1)
    stable_order = np.argsort(~keep_mask, axis=1, kind="stable")
    compact_ids = np.take_along_axis(gt_ids_new, stable_order, axis=1)
    compact_dists = np.take_along_axis(gt_dists, stable_order, axis=1)
    valid_slots = np.arange(topk)[None, :] < kept_counts[:, None]
    out_ids = np.where(valid_slots, compact_ids, UINT32_MAX).astype(np.uint32, copy=False)
    out_dists = np.where(valid_slots, compact_dists, np.float32(np.inf)).astype(np.float32, copy=False)
    return out_ids, out_dists


def write_readme(path: Path) -> None:
    path.write_text(
        "# SIFT10M Update Eval UInt8\n\n"
        "This dataset is derived from the first 10,000,000 vectors of the TexMex "
        "ANN_SIFT1B corpus. The TexMex source files are named `bigann_*.bvecs`; "
        "the vectors are stored here as uint8 `.u8bin` artifacts, not float32.\n\n"
        "Layout: 10M total vectors, 8M initial active vectors, 100k updates per "
        "batch, 100 ring update batches plus batch 0000 initial state, seed 42. "
        "The HNSW index is native uint8 with M=32 and ef_construction=500. "
        "PQ artifacts use m16/ks256, giving 16-byte codes.\n",
        encoding="utf-8",
    )


def write_kaggle_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "title": "SIFT10M Update Eval UInt8",
                "id": "shurangwu/sift10m-update-eval-uint8",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_u8bin_path)
    query_path = Path(args.query_u8bin_path)
    gt_path = Path(args.groundtruth_path)
    out_dir = Path(args.output_dir)
    for path in (base_path, query_path, gt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    base_header = read_u8bin_header(base_path)
    query_header = read_u8bin_header(query_path)
    if base_header.count != args.total_vectors or base_header.dim != 128:
        raise ValueError(f"unexpected base header: {base_header}")
    if query_header.count != 10_000 or query_header.dim != base_header.dim:
        raise ValueError(f"unexpected query header: {query_header}")

    gt_header, gt_ids_old, gt_dists = read_groundtruth(gt_path)
    if gt_header.num_queries != query_header.count:
        raise ValueError("query/groundtruth count mismatch")

    permutation = build_permutation(args.total_vectors, args.seed)
    inverse_permutation = np.empty(args.total_vectors, dtype=np.uint32)
    inverse_permutation[permutation] = np.arange(args.total_vectors, dtype=np.uint32)
    gt_ids_new = remap_groundtruth_ids(gt_ids_old, inverse_permutation, args.total_vectors)

    if args.dry_run:
        print("dry_run ok")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_readme(out_dir / "README.md")
    write_kaggle_metadata(out_dir / "dataset-metadata.json")
    np.save(out_dir / "permutation.npy", permutation, allow_pickle=False)
    write_ids_bin(out_dir / "initial_active_ids.bin", np.arange(args.initial_active, dtype=np.uint32))
    place_file(base_path, out_dir / "base.u8bin", args.base_link_mode, args.force)
    place_file(query_path, out_dir / "queries" / "unif_query_10k.u8bin", args.base_link_mode, args.force)
    materialize_permuted_u8bin(
        input_path=base_path,
        output_path=out_dir / "base_permuted.u8bin",
        expected_count=args.total_vectors,
        dim=base_header.dim,
        permutation=permutation,
        chunk_rows=args.chunk_rows,
        force=args.force,
    )

    batch_entries = []
    for batch_idx in tqdm(range(args.num_batches + 1), desc="Writing batch workload", unit="batch"):
        batch = build_batch_spec(batch_idx, args.initial_active, args.batch_size, args.total_vectors)
        gt_ids_filtered, gt_dists_filtered = filter_groundtruth_for_active_segments(
            gt_ids_new,
            gt_dists,
            batch.active_segments,
        )
        batch_name = f"batch_{batch_idx:04d}"
        gt_path_out = out_dir / "groundtruth" / f"{batch_name}_unif_groundtruth_10k.bin"
        query_rel = "queries/unif_query_10k.u8bin"
        gt_rel = relative(gt_path_out, out_dir)
        batch_json_path = out_dir / "batches" / f"{batch_name}.json"
        batch_rel = relative(batch_json_path, out_dir)
        write_groundtruth(gt_path_out, gt_ids_filtered, gt_dists_filtered)
        ensure_parent(batch_json_path)
        batch_json_path.write_text(
            json.dumps(
                batch_json_object(batch, args.initial_active, gt_rel, query_rel),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        batch_entries.append(batch_manifest_entry(batch, args.initial_active, gt_rel, query_rel, batch_rel))

    manifest = {
        "dataset": "sift10m-update-eval-uint8",
        "source": {
            "description": "TexMex ANN_SIFT1B first 10M vectors; source files are named bigann_*.bvecs.",
            "base_source": str(base_path),
            "query_source": str(query_path),
            "groundtruth_source": str(gt_path),
        },
        "id_semantics": {
            "description": "All IDs are post-permutation external IDs.",
            "mapping": "permutation[new_external_id] = original_sift1b_bigann_row_id",
        },
        "inputs": {
            "base_u8bin_path": "base.u8bin",
            "query_u8bin_path": "queries/unif_query_10k.u8bin",
            "groundtruth_path": str(gt_path),
            "base_count": base_header.count,
            "base_dim": base_header.dim,
            "query_count": query_header.count,
            "query_dim": query_header.dim,
            "groundtruth_num_queries": gt_header.num_queries,
            "groundtruth_topk": gt_header.topk,
        },
        "config": {
            "total_vectors": args.total_vectors,
            "initial_active": args.initial_active,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "includes_batch_zero": True,
            "seed": args.seed,
            "window_mode": "ring",
            "vector_dtype": "uint8",
            "single_query_artifact": True,
            "materialize_permuted_base": True,
        },
        "artifacts": {
            "output_dir": ".",
            "permutation_path": "permutation.npy",
            "initial_active_ids_path": "initial_active_ids.bin",
            "base_permuted_path": "base_permuted.u8bin",
            "index_path": "index_m_32_ef_500_initial_8m",
            "index_meta_path": "index_m_32_ef_500_initial_8m.hnswmeta.json",
            "pq": {
                "enabled": True,
                "variant": "m16",
                "source_order": "original_sift_row",
                "runtime_order": "initial_index_internal",
                "codes_path": "pq/codes_initial_8000000_m16.pqcodes",
                "meta_path": "pq/meta_m16.pqmeta",
                "codebook_path": "pq/codebook_m16.pqcodebook",
            },
        },
        "batches": batch_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote workload: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
