// Build with vectordb-cxl's quantization library, for example:
// g++ -std=c++20 -O3 -I/data/vectordb-cxl/src \
//   -I/data/vectordb-cxl/build/_deps/faiss-src \
//   -I/data/vectordb-cxl/build/_deps/faiss-build \
//   examples/cpp/generate_u8bin_m16_pq.cpp \
//   /data/vectordb-cxl/build/src/quantization/libvectordb_quantization.a \
//   /data/vectordb-cxl/build/_deps/faiss-build/faiss/libfaiss.a \
//   -fopenmp -lblas -llapack -o /tmp/generate_u8bin_m16_pq

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/configuration.h"
#include "quantization/pq_quantizer.h"

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path input_u8bin;
    fs::path output_dir;
    std::uint32_t m{16};
    std::uint32_t ks{256};
    std::uint32_t training_iterations{20};
    std::uint32_t seed{42};
    std::size_t batch_rows{100000};
};

template <typename T>
void read_exact(std::istream& in, T& value, const char* name) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in.good()) {
        throw std::runtime_error(std::string("failed to read ") + name);
    }
}

template <typename T>
void write_value(std::ostream& out, T value, const char* name) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out.good()) {
        throw std::runtime_error(std::string("failed to write ") + name);
    }
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (opt == "--input-u8bin") {
            args.input_u8bin = next("--input-u8bin");
        } else if (opt == "--output-dir") {
            args.output_dir = next("--output-dir");
        } else if (opt == "--m") {
            args.m = static_cast<std::uint32_t>(std::stoul(next("--m")));
        } else if (opt == "--ks") {
            args.ks = static_cast<std::uint32_t>(std::stoul(next("--ks")));
        } else if (opt == "--training-iterations") {
            args.training_iterations =
                static_cast<std::uint32_t>(std::stoul(next("--training-iterations")));
        } else if (opt == "--seed") {
            args.seed = static_cast<std::uint32_t>(std::stoul(next("--seed")));
        } else if (opt == "--batch-rows") {
            args.batch_rows = static_cast<std::size_t>(std::stoull(next("--batch-rows")));
        } else if (opt == "--help" || opt == "-h") {
            std::cout
                << "Usage: generate_u8bin_m16_pq --input-u8bin PATH --output-dir DIR "
                << "[--m 16] [--ks 256] [--training-iterations 20] [--seed 42] "
                << "[--batch-rows 100000]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + opt);
        }
    }
    if (args.input_u8bin.empty()) {
        throw std::invalid_argument("--input-u8bin is required");
    }
    if (args.output_dir.empty()) {
        throw std::invalid_argument("--output-dir is required");
    }
    if (args.batch_rows == 0) {
        throw std::invalid_argument("--batch-rows must be > 0");
    }
    return args;
}

std::vector<float> load_u8bin_as_float(const fs::path& path, std::uint32_t& count, std::uint32_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open input: " + path.string());
    }
    read_exact(in, count, "u8bin count");
    read_exact(in, dim, "u8bin dim");
    if (count == 0 || dim == 0) {
        throw std::runtime_error("invalid u8bin shape");
    }

    const auto expected_size = static_cast<std::uintmax_t>(8) +
                               static_cast<std::uintmax_t>(count) * dim;
    const auto actual_size = fs::file_size(path);
    if (actual_size != expected_size) {
        throw std::runtime_error("u8bin file size mismatch");
    }

    std::vector<float> vectors(static_cast<std::size_t>(count) * dim);
    std::vector<std::uint8_t> row(dim);
    for (std::uint32_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(dim));
        if (!in.good()) {
            throw std::runtime_error("truncated u8bin payload");
        }
        float* dst = vectors.data() + static_cast<std::size_t>(i) * dim;
        for (std::uint32_t d = 0; d < dim; ++d) {
            dst[d] = static_cast<float>(row[d]);
        }
        if ((i + 1) % 1000000 == 0) {
            std::cerr << "loaded " << (i + 1) << "/" << count << " vectors\n";
        }
    }
    return vectors;
}

void save_codes_streaming(const vectordb::PQQuantizer& quantizer,
                          const vectordb::QuantizationConfig& cfg,
                          const float* vectors,
                          std::uint64_t count,
                          std::size_t batch_rows,
                          const fs::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open pqcodes output: " + path.string());
    }
    const std::uint32_t ds = cfg.dim / cfg.m;
    const std::uint8_t metric_l2 = 0;
    const std::uint32_t code_dtype_bytes = quantizer.code_dtype_bytes();
    write_value(out, count, "pq codes num_vectors");
    write_value(out, cfg.dim, "pq codes dim");
    write_value(out, cfg.m, "pq codes m");
    write_value(out, cfg.ks, "pq codes ks");
    write_value(out, ds, "pq codes ds");
    write_value(out, metric_l2, "pq codes metric");
    write_value(out, code_dtype_bytes, "pq codes code dtype bytes");

    const std::size_t code_bytes = quantizer.bytes_per_code();
    std::vector<std::uint8_t> codes(batch_rows * code_bytes);
    for (std::uint64_t start = 0; start < count; start += batch_rows) {
        const auto rows =
            static_cast<std::size_t>(std::min<std::uint64_t>(batch_rows, count - start));
        quantizer.encode_batch(vectors + start * cfg.dim, rows, codes.data());
        out.write(reinterpret_cast<const char*>(codes.data()),
                  static_cast<std::streamsize>(rows * code_bytes));
        if (!out.good()) {
            throw std::runtime_error("failed while writing pqcodes payload");
        }
        std::cerr << "encoded " << (start + rows) << "/" << count << " vectors\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        std::uint32_t count = 0;
        std::uint32_t dim = 0;
        std::vector<float> vectors = load_u8bin_as_float(args.input_u8bin, count, dim);

        vectordb::QuantizationConfig cfg;
        cfg.method = vectordb::QuantizationMethod::PQ;
        cfg.dim = dim;
        cfg.metric = vectordb::QuantizationMetric::L2;
        cfg.training_iterations = args.training_iterations;
        cfg.seed = args.seed;
        cfg.m = args.m;
        cfg.ks = args.ks;
        cfg.save_codes = false;

        vectordb::PQQuantizer quantizer(cfg);
        std::cerr << "training PQ on " << count << " vectors, dim=" << dim
                  << ", m=" << args.m << ", ks=" << args.ks << "\n";
        quantizer.fit(vectors.data(), count);

        fs::create_directories(args.output_dir);
        quantizer.save_metadata((args.output_dir / "meta_m16.pqmeta").string());
        quantizer.save_codebook((args.output_dir / "codebook_m16.pqcodebook").string());
        save_codes_streaming(quantizer,
                             cfg,
                             vectors.data(),
                             count,
                             args.batch_rows,
                             args.output_dir / "codes_full_10000000_m16_original_order.pqcodes");
        std::cerr << "wrote PQ artifacts under " << args.output_dir << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
