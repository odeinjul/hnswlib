import argparse
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass

import numpy as np


HEADER_STRUCT = struct.Struct("<QQQQQQiIQQQdQ")
LABEL_STRUCT = struct.Struct("<Q")
UINT32_STRUCT = struct.Struct("<I")


@dataclass
class HnswHeader:
    offset_level0: int
    max_elements: int
    cur_element_count: int
    size_data_per_element: int
    label_offset: int
    offset_data: int
    maxlevel: int
    enterpoint_node: int
    max_M: int
    max_M0: int
    M: int
    mult: float
    ef_construction: int

    @classmethod
    def read(cls, input_file):
        data = input_file.read(HEADER_STRUCT.size)
        if len(data) != HEADER_STRUCT.size:
            raise RuntimeError("HNSW index is missing the fixed header")
        return cls(*HEADER_STRUCT.unpack(data))

    def pack(self):
        return HEADER_STRUCT.pack(
            self.offset_level0,
            self.max_elements,
            self.cur_element_count,
            self.size_data_per_element,
            self.label_offset,
            self.offset_data,
            self.maxlevel,
            self.enterpoint_node,
            self.max_M,
            self.max_M0,
            self.M,
            self.mult,
            self.ef_construction,
        )

    def with_uint8_payload(self, dim):
        label_offset = self.offset_data + dim
        return HnswHeader(
            self.offset_level0,
            self.max_elements,
            self.cur_element_count,
            label_offset + LABEL_STRUCT.size,
            label_offset,
            self.offset_data,
            self.maxlevel,
            self.enterpoint_node,
            self.max_M,
            self.max_M0,
            self.M,
            self.mult,
            self.ef_construction,
        )


@dataclass
class BaseVectors:
    path: str
    count: int
    dim: int
    dtype: np.dtype
    vectors: np.memmap


def read_vector_file_header(path, dtype, format_name):
    with open(path, "rb") as input_file:
        header = np.fromfile(input_file, dtype=np.int32, count=2)
    if header.size != 2:
        raise RuntimeError("%s file is missing the two-int32 header" % format_name)
    count, dim = int(header[0]), int(header[1])
    if count < 0 or dim <= 0:
        raise RuntimeError("%s header has invalid shape" % format_name)
    expected_size = 8 + count * dim * np.dtype(dtype).itemsize
    actual_size = os.path.getsize(path)
    if actual_size < expected_size:
        raise RuntimeError("%s file is shorter than its header shape" % format_name)
    return count, dim


def read_base_vectors(path):
    if path.endswith(".u8bin"):
        dtype = np.dtype(np.uint8)
        format_name = "u8bin"
    elif path.endswith(".fbin"):
        dtype = np.dtype(np.float32)
        format_name = "fbin"
    else:
        raise RuntimeError("Base vectors must be a BigANN-style .u8bin or .fbin file")

    count, dim = read_vector_file_header(path, dtype, format_name)
    vectors = np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(count, dim))
    return BaseVectors(path=path, count=count, dim=dim, dtype=dtype, vectors=vectors)


def rows_to_uint8(rows):
    if rows.dtype == np.uint8:
        return np.asarray(rows)
    if rows.dtype != np.float32:
        raise RuntimeError("Unsupported base vector dtype for uint8 conversion")
    if not np.all(np.isfinite(rows)):
        raise RuntimeError("Float base vectors contain non-finite values")
    rounded = np.rint(rows)
    if not np.array_equal(rows, rounded):
        raise RuntimeError("Float base vectors contain non-integer values and cannot be converted exactly to uint8")
    if np.any(rounded < 0) or np.any(rounded > 255):
        raise RuntimeError("Float base vectors contain values outside uint8 range")
    return rounded.astype(np.uint8)


def rows_as_float32(rows):
    if rows.dtype == np.float32:
        return np.asarray(rows)
    if rows.dtype == np.uint8:
        return rows.astype(np.float32)
    raise RuntimeError("Unsupported base vector dtype for float32 comparison")


def validate_source_header(header, dim, base_count, source_size):
    expected_offset_data = UINT32_STRUCT.size + header.max_M0 * UINT32_STRUCT.size
    if header.offset_level0 != 0:
        raise RuntimeError("Only HNSW indexes with offsetLevel0=0 are supported")
    if header.offset_data != expected_offset_data:
        raise RuntimeError("HNSW offsetData does not match max_M0 layout")
    if header.label_offset < header.offset_data:
        raise RuntimeError("HNSW label offset precedes vector data")
    if header.size_data_per_element != header.label_offset + LABEL_STRUCT.size:
        raise RuntimeError("HNSW row size does not match label offset")
    if header.cur_element_count > header.max_elements:
        raise RuntimeError("HNSW current element count exceeds max elements")
    if header.cur_element_count > base_count:
        raise RuntimeError("HNSW index has more elements than the base vector file")

    level0_end = HEADER_STRUCT.size + header.cur_element_count * header.size_data_per_element
    if level0_end > source_size:
        raise RuntimeError("HNSW level-0 block extends past the source file")

    payload_bytes = header.label_offset - header.offset_data
    if payload_bytes not in (dim, dim * np.dtype(np.float32).itemsize):
        raise RuntimeError(
            "Unsupported vector payload byte width: %d; expected %d for uint8 or %d for float32"
            % (payload_bytes, dim, dim * np.dtype(np.float32).itemsize)
        )
    return payload_bytes


def iter_validation_ids(count, sample_count):
    if sample_count <= 0 or count == 0:
        return []
    sample_count = min(sample_count, count)
    if sample_count == count:
        return list(range(count))
    return sorted(set(int(x) for x in np.linspace(0, count - 1, sample_count)))


def validate_rows(source_path, header, base, payload_bytes, sample_count):
    bad = []
    with open(source_path, "rb") as input_file:
        for internal_id in iter_validation_ids(header.cur_element_count, sample_count):
            row_offset = HEADER_STRUCT.size + internal_id * header.size_data_per_element
            input_file.seek(row_offset + header.offset_data)
            payload = input_file.read(payload_bytes)
            input_file.seek(row_offset + header.label_offset)
            label = LABEL_STRUCT.unpack(input_file.read(LABEL_STRUCT.size))[0]
            if label >= base.count:
                bad.append((internal_id, label, "label out of range"))
                continue
            expected = np.asarray(base.vectors[label])
            if payload_bytes == base.dim:
                matches = payload == rows_to_uint8(expected).tobytes()
            else:
                vector = np.frombuffer(payload, dtype=np.float32, count=base.dim)
                matches = np.array_equal(vector, rows_as_float32(expected))
            if not matches:
                bad.append((internal_id, label, "payload does not match base[label]"))
            if bad:
                break
    if bad:
        internal_id, label, reason = bad[0]
        raise RuntimeError("Source validation failed at internal_id=%d label=%d: %s" % (internal_id, label, reason))


def write_metadata(output_path, header, dim):
    metadata = {
        "format": "hnswlib-typed-index-metadata",
        "version": 1,
        "space": "l2",
        "dtype": "uint8",
        "dim": dim,
        "distance_type": "float32",
        "payload_bytes_per_vector": dim,
        "index_file": output_path,
        "index_file_size": os.path.getsize(output_path),
        "max_elements": header.max_elements,
        "cur_element_count": header.cur_element_count,
        "size_data_per_element": header.size_data_per_element,
        "label_offset": header.label_offset,
        "offset_data": header.offset_data,
        "max_M": header.max_M,
        "max_M0": header.max_M0,
        "M": header.M,
        "ef_construction": header.ef_construction,
    }
    with open(output_path + ".hnswmeta.json", "w") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def convert_index(source_index, base_path, output_index, verify_samples=1024, chunk_rows=100000):
    base = read_base_vectors(base_path)
    source_size = os.path.getsize(source_index)

    with open(source_index, "rb") as source_file:
        source_header = HnswHeader.read(source_file)

    payload_bytes = validate_source_header(source_header, base.dim, base.count, source_size)
    validate_rows(source_index, source_header, base, payload_bytes, verify_samples)

    os.makedirs(os.path.dirname(os.path.abspath(output_index)), exist_ok=True)
    output_header = source_header.with_uint8_payload(base.dim)
    same_index_path = os.path.abspath(source_index) == os.path.abspath(output_index)
    start = time.time()
    rows_verified_during_conversion = 0

    if payload_bytes == base.dim:
        if not same_index_path:
            shutil.copyfile(source_index, output_index)
    else:
        if same_index_path:
            raise RuntimeError("In-place float32-to-uint8 conversion is not supported; write to a new index path")
        with open(source_index, "rb") as source_file, open(output_index, "wb") as output_file:
            source_file.seek(HEADER_STRUCT.size)
            output_file.write(output_header.pack())

            remaining = source_header.cur_element_count
            while remaining:
                rows = min(remaining, chunk_rows)
                block = source_file.read(rows * source_header.size_data_per_element)
                if len(block) != rows * source_header.size_data_per_element:
                    raise RuntimeError("HNSW source ended inside the level-0 block")

                source_rows = np.frombuffer(block, dtype=np.uint8).reshape(rows, source_header.size_data_per_element)
                labels = np.ndarray(
                    shape=(rows,),
                    dtype="<u8",
                    buffer=block,
                    offset=source_header.label_offset,
                    strides=(source_header.size_data_per_element,),
                ).copy()
                if labels.size and int(labels.max()) >= base.count:
                    raise RuntimeError("HNSW label %d is outside the base vector file" % int(labels.max()))
                index_labels = labels.astype(np.intp, copy=False)

                expected_rows = np.asarray(base.vectors[index_labels])
                source_payload = np.ndarray(
                    shape=(rows, base.dim),
                    dtype="<f4",
                    buffer=block,
                    offset=source_header.offset_data,
                    strides=(source_header.size_data_per_element, np.dtype(np.float32).itemsize),
                )
                expected_float_rows = rows_as_float32(expected_rows)
                if not np.array_equal(source_payload, expected_float_rows):
                    mismatch = np.argwhere(source_payload != expected_float_rows)[0]
                    bad_row = int(mismatch[0])
                    raise RuntimeError(
                        "Source validation failed at internal_id=%d label=%d: float payload does not match base[label]"
                        % (source_header.cur_element_count - remaining + bad_row, int(labels[bad_row]))
                    )

                output_rows = np.empty((rows, output_header.size_data_per_element), dtype=np.uint8)
                output_rows[:, : source_header.offset_data] = source_rows[:, : source_header.offset_data]
                output_rows[:, output_header.offset_data : output_header.label_offset] = rows_to_uint8(expected_rows)
                output_rows[:, output_header.label_offset : output_header.label_offset + LABEL_STRUCT.size] = source_rows[
                    :, source_header.label_offset : source_header.label_offset + LABEL_STRUCT.size
                ]
                output_file.write(output_rows.tobytes())
                rows_verified_during_conversion += rows
                remaining -= rows

            shutil.copyfileobj(source_file, output_file, length=16 * 1024 * 1024)

    write_metadata(output_index, output_header, base.dim)
    elapsed = time.time() - start
    return {
        "source_index": source_index,
        "output_index": output_index,
        "base_vectors": base_path,
        "base_dtype": str(base.dtype),
        "dim": base.dim,
        "vectors": source_header.cur_element_count,
        "source_payload_bytes": payload_bytes,
        "output_payload_bytes": base.dim,
        "source_index_file_size": source_size,
        "output_index_file_size": os.path.getsize(output_index),
        "metadata_file": output_index + ".hnswmeta.json",
        "conversion_seconds": elapsed,
        "rows_verified_during_conversion": rows_verified_during_conversion,
        "mode": "copied_uint8_payload" if payload_bytes == base.dim else "converted_float32_payload",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert or tag a BigANN HNSW index so it loads as a native uint8 hnswlib index."
    )
    parser.add_argument("source_index", help="Existing HNSW index to read")
    parser.add_argument("base_vectors", help="BigANN .u8bin or .fbin base vectors used to build the index")
    parser.add_argument("output_index", help="Converted native uint8 HNSW index path")
    parser.add_argument("--verify-samples", type=int, default=1024)
    parser.add_argument("--chunk-rows", type=int, default=100000)
    args = parser.parse_args()

    result = convert_index(
        args.source_index,
        args.base_vectors,
        args.output_index,
        verify_samples=args.verify_samples,
        chunk_rows=args.chunk_rows,
    )
    for key in sorted(result):
        value = result[key]
        if isinstance(value, float):
            value = "%.3f" % value
        print("%s: %s" % (key, value))


if __name__ == "__main__":
    main()
