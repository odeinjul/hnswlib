#!/usr/bin/env python3

import argparse
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


PQ_CODES_HEADER_BYTES = 29
HNSW_DISK_HEADER_FORMAT = "<6QiI3QdQ"
HNSW_DISK_HEADER_BYTES = struct.calcsize(HNSW_DISK_HEADER_FORMAT)


@dataclass(frozen=True)
class PqCodesHeader:
    num_vectors: int
    dim: int
    m: int
    ks: int
    ds: int
    metric: int
    code_dtype_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize SIFT10M m16 PQ artifacts: document full 10M original-order "
            "PQ and write initial-8M HNSW-internal-order runtime PQ."
        )
    )
    parser.add_argument(
        "--pq-dir",
        default="/data/vectordb-cxl/data/sift10m-pq-m16",
    )
    parser.add_argument(
        "--update-dir",
        default="/data/vectordb-cxl/data/sift10m-update-eval-uint8",
    )
    parser.add_argument("--initial-active", type=int, default=8_000_000)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.initial_active <= 0:
        parser.error("--initial-active must be > 0")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be > 0")
    return args


def parse_pq_codes_header(path: Path) -> PqCodesHeader:
    with path.open("rb") as f:
        raw = f.read(PQ_CODES_HEADER_BYTES)
    if len(raw) != PQ_CODES_HEADER_BYTES:
        raise ValueError(f"invalid pqcodes header in {path}")
    return PqCodesHeader(
        num_vectors=struct.unpack_from("<Q", raw, 0)[0],
        dim=struct.unpack_from("<I", raw, 8)[0],
        m=struct.unpack_from("<I", raw, 12)[0],
        ks=struct.unpack_from("<I", raw, 16)[0],
        ds=struct.unpack_from("<I", raw, 20)[0],
        metric=raw[24],
        code_dtype_bytes=struct.unpack_from("<I", raw, 25)[0],
    )


def pq_code_bytes(header: PqCodesHeader) -> int:
    if header.ks == 256 and header.code_dtype_bytes == 1:
        return header.m
    raise ValueError(f"unsupported PQ packing: {header}")


def pq_header_to_bytes(header: PqCodesHeader) -> bytes:
    return (
        struct.pack("<Q", header.num_vectors)
        + struct.pack("<I", header.dim)
        + struct.pack("<I", header.m)
        + struct.pack("<I", header.ks)
        + struct.pack("<I", header.ds)
        + struct.pack("<B", header.metric)
        + struct.pack("<I", header.code_dtype_bytes)
    )


def read_hnsw_external_labels(path: Path, expected_count: int) -> np.ndarray:
    with path.open("rb") as f:
        raw = f.read(HNSW_DISK_HEADER_BYTES)
        if len(raw) != HNSW_DISK_HEADER_BYTES:
            raise ValueError(f"invalid HNSW header in {path}")
        (
            _offset_level0,
            _max_elements,
            cur_element_count,
            size_data_per_element,
            label_offset,
            _offset_data,
            _max_level,
            _enterpoint_node,
            _max_m,
            _max_m0,
            _m,
            _mult,
            _ef_construction,
        ) = struct.unpack(HNSW_DISK_HEADER_FORMAT, raw)
        if cur_element_count != expected_count:
            raise ValueError(f"HNSW label count mismatch: {cur_element_count} != {expected_count}")
        labels = np.empty(expected_count, dtype=np.uint32)
        for internal_id in tqdm(range(expected_count), desc="Reading HNSW labels", unit="label"):
            f.seek(HNSW_DISK_HEADER_BYTES + internal_id * size_data_per_element + label_offset)
            label_raw = f.read(8)
            if len(label_raw) != 8:
                raise ValueError(f"truncated label row {internal_id}")
            label = struct.unpack("<Q", label_raw)[0]
            if label >= expected_count:
                raise ValueError(f"label {label} out of range")
            labels[internal_id] = np.uint32(label)
    if np.unique(labels).size != expected_count:
        raise ValueError("HNSW labels are not a permutation")
    return labels


def write_runtime_pq(
    full_codes_path: Path,
    permutation_path: Path,
    index_path: Path,
    output_codes_path: Path,
    initial_active: int,
    chunk_rows: int,
    force: bool,
) -> None:
    header = parse_pq_codes_header(full_codes_path)
    if header.num_vectors != 10_000_000 or header.dim != 128 or header.m != 16 or header.ks != 256 or header.ds != 8:
        raise ValueError(f"unexpected full PQ header: {header}")
    code_bytes = pq_code_bytes(header)
    expected_size = PQ_CODES_HEADER_BYTES + header.num_vectors * code_bytes
    if full_codes_path.stat().st_size != expected_size:
        raise ValueError("full PQ file size mismatch")

    permutation = np.load(permutation_path, allow_pickle=False)
    if permutation.dtype != np.uint32 or permutation.shape != (header.num_vectors,):
        raise ValueError("permutation dtype/shape mismatch")

    labels = read_hnsw_external_labels(index_path, initial_active)
    source_rows = permutation[labels]

    if output_codes_path.exists():
        if not force:
            raise FileExistsError(f"{output_codes_path} exists; pass --force")
        output_codes_path.unlink()
    output_codes_path.parent.mkdir(parents=True, exist_ok=True)

    payload = np.memmap(
        full_codes_path,
        dtype=np.uint8,
        mode="r",
        offset=PQ_CODES_HEADER_BYTES,
        shape=(header.num_vectors, code_bytes),
    )
    out_header = PqCodesHeader(
        num_vectors=initial_active,
        dim=header.dim,
        m=header.m,
        ks=header.ks,
        ds=header.ds,
        metric=header.metric,
        code_dtype_bytes=header.code_dtype_bytes,
    )
    with output_codes_path.open("wb") as out:
        out.write(pq_header_to_bytes(out_header))
        for start in tqdm(range(0, initial_active, chunk_rows), desc="Writing runtime PQ", unit="chunk"):
            end = min(start + chunk_rows, initial_active)
            rows = payload[source_rows[start:end]]
            np.asarray(rows, dtype=np.uint8).tofile(out)


def write_pq_dataset_docs(pq_dir: Path) -> None:
    (pq_dir / "README.md").write_text(
        "# SIFT10M m16 PQ\n\n"
        "Full 10M m16/ks256 PQ artifacts for the TexMex ANN_SIFT1B first 10M "
        "vectors. The source file names in TexMex are `bigann_*.bvecs`; codes "
        "are stored in original SIFT row order. Each vector has a 16-byte code.\n",
        encoding="utf-8",
    )
    (pq_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "sift10m-pq-m16",
                "source": "TexMex ANN_SIFT1B first 10M vectors, stored from bigann_base.bvecs as uint8",
                "order": "original_sift_row_order",
                "pq": {
                    "dim": 128,
                    "m": 16,
                    "ks": 256,
                    "ds": 8,
                    "code_dtype_bytes": 1,
                    "code_bytes_per_vector": 16,
                    "num_vectors": 10_000_000,
                    "codes_path": "codes_full_10000000_m16_original_order.pqcodes",
                    "meta_path": "meta_m16.pqmeta",
                    "codebook_path": "codebook_m16.pqcodebook",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (pq_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SIFT10M PQ m16",
                "id": "shurangwu/sift10m-pq-m16",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    pq_dir = Path(args.pq_dir)
    update_dir = Path(args.update_dir)
    full_codes_path = pq_dir / "codes_full_10000000_m16_original_order.pqcodes"
    meta_path = pq_dir / "meta_m16.pqmeta"
    codebook_path = pq_dir / "codebook_m16.pqcodebook"
    for path in (full_codes_path, meta_path, codebook_path):
        if not path.exists():
            raise FileNotFoundError(path)

    update_pq_dir = update_dir / "pq"
    update_pq_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_path, update_pq_dir / "meta_m16.pqmeta")
    shutil.copy2(codebook_path, update_pq_dir / "codebook_m16.pqcodebook")
    write_runtime_pq(
        full_codes_path=full_codes_path,
        permutation_path=update_dir / "permutation.npy",
        index_path=update_dir / "index_m_32_ef_500_initial_8m",
        output_codes_path=update_pq_dir / "codes_initial_8000000_m16.pqcodes",
        initial_active=args.initial_active,
        chunk_rows=args.chunk_rows,
        force=args.force,
    )
    write_pq_dataset_docs(pq_dir)
    print(f"wrote runtime PQ: {update_pq_dir / 'codes_initial_8000000_m16.pqcodes'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
