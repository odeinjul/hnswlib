#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <type_traits>

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


inline void assert_true(bool expr, const std::string & msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}


inline bool file_exists(const std::string &path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    return input.is_open();
}


inline std::string read_text_file(const std::string &path) {
    std::ifstream input(path.c_str());
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}


inline std::string json_get_string(const std::string &json, const std::string &key) {
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(json, match, re)) {
        throw std::runtime_error("Typed HNSW metadata missing string field: " + key);
    }
    return match[1].str();
}


inline size_t json_get_size_t(const std::string &json, const std::string &key) {
    std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(json, match, re)) {
        throw std::runtime_error("Typed HNSW metadata missing integer field: " + key);
    }
    return (size_t) std::stoull(match[1].str());
}


inline std::string json_escape(const std::string &value) {
    std::string escaped;
    for (size_t i = 0; i < value.size(); i++) {
        char c = value[i];
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return escaped;
}


struct HnswHeaderInfo {
    size_t offset_level0;
    size_t max_elements;
    size_t cur_element_count;
    size_t size_data_per_element;
    size_t label_offset;
    size_t offset_data;
    int max_level;
    hnswlib::tableint enterpoint_node;
    size_t max_M;
    size_t max_M0;
    size_t M;
    double mult;
    size_t ef_construction;
    size_t file_size;
};


inline HnswHeaderInfo read_hnsw_header(const std::string &path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    HnswHeaderInfo header;
    input.seekg(0, input.end);
    header.file_size = (size_t) input.tellg();
    input.seekg(0, input.beg);

    hnswlib::readBinaryPOD(input, header.offset_level0);
    hnswlib::readBinaryPOD(input, header.max_elements);
    hnswlib::readBinaryPOD(input, header.cur_element_count);
    hnswlib::readBinaryPOD(input, header.size_data_per_element);
    hnswlib::readBinaryPOD(input, header.label_offset);
    hnswlib::readBinaryPOD(input, header.offset_data);
    hnswlib::readBinaryPOD(input, header.max_level);
    hnswlib::readBinaryPOD(input, header.enterpoint_node);
    hnswlib::readBinaryPOD(input, header.max_M);
    hnswlib::readBinaryPOD(input, header.max_M0);
    hnswlib::readBinaryPOD(input, header.M);
    hnswlib::readBinaryPOD(input, header.mult);
    hnswlib::readBinaryPOD(input, header.ef_construction);

    return header;
}


class CustomFilterFunctor: public hnswlib::BaseFilterFunctor {
    std::function<bool(hnswlib::labeltype)> filter;

 public:
    explicit CustomFilterFunctor(const std::function<bool(hnswlib::labeltype)>& f) {
        filter = f;
    }

    bool operator()(hnswlib::labeltype id) {
        return filter(id);
    }
};


inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.",
            buffer.ndim);
        throw std::runtime_error(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}


inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
              (ids_numpy.ndim == 0 && feature_rows == 1))) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "The input label shape %d does not match the input data vector shape %d",
                ids_numpy.ndim, feature_rows);
            throw std::runtime_error(msg);
        }
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}


template<typename data_t>
struct IndexDType {
    static std::string name() {
        return "float32";
    }
};

template<>
struct IndexDType<uint8_t> {
    static std::string name() {
        return "uint8";
    }
};

template<>
struct IndexDType<int8_t> {
    static std::string name() {
        return "int8";
    }
};


template<typename data_t>
struct DataArrayCaster {
    typedef py::array_t<data_t, py::array::c_style | py::array::forcecast> Array;

    static Array cast(py::object input) {
        return Array(input);
    }
};

template<>
struct DataArrayCaster<uint8_t> {
    typedef py::array_t<uint8_t, py::array::c_style> Array;

    static Array cast(py::object input) {
        py::array array = py::array::ensure(input);
        if (!array || array.request().format != py::format_descriptor<uint8_t>::format()) {
            throw std::runtime_error("Input vector dtype must match index dtype uint8.");
        }
        return Array(input);
    }
};

template<>
struct DataArrayCaster<int8_t> {
    typedef py::array_t<int8_t, py::array::c_style> Array;

    static Array cast(py::object input) {
        py::array array = py::array::ensure(input);
        if (!array || array.request().format != py::format_descriptor<int8_t>::format()) {
            throw std::runtime_error("Input vector dtype must match index dtype int8.");
        }
        return Array(input);
    }
};


template<typename dist_t, typename data_t = float>
class TypedIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;


    TypedIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            if (std::is_same<data_t, float>::value) {
                l2space = new hnswlib::L2Space(dim);
            } else if (std::is_same<data_t, uint8_t>::value) {
                l2space = new hnswlib::L2SpaceUInt8(dim);
            } else if (std::is_same<data_t, int8_t>::value) {
                l2space = new hnswlib::L2SpaceInt8(dim);
            } else {
                throw std::runtime_error("Unsupported index dtype.");
            }
        } else if (space_name == "ip") {
            if (!std::is_same<data_t, float>::value)
                throw std::runtime_error("Integer dtypes are supported only for space='l2'.");
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            if (!std::is_same<data_t, float>::value)
                throw std::runtime_error("Integer dtypes are supported only for space='l2'.");
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~TypedIndex() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }

    void setSearchAccessMetricsEnabled(bool enabled) {
        if (!appr_alg)
            throw std::runtime_error("The index is not initialized.");
        appr_alg->setSearchAccessMetricsEnabled(enabled);
    }

    void resetSearchAccessMetrics() {
        if (!appr_alg)
            throw std::runtime_error("The index is not initialized.");
        appr_alg->resetSearchAccessMetrics();
    }

    py::dict getSearchAccessMetrics() const {
        if (!appr_alg)
            throw std::runtime_error("The index is not initialized.");
        const hnswlib::SearchAccessMetrics metrics =
            appr_alg->getSearchAccessMetrics();
        py::dict result;
        result["enabled"] = appr_alg->searchAccessMetricsEnabled();
        result["entrypoint_vector_accesses"] =
            metrics.entrypoint_vector_accesses;
        result["upper_neighbor_list_accesses"] =
            metrics.upper_neighbor_list_accesses;
        result["upper_vector_accesses"] = metrics.upper_vector_accesses;
        result["l0_neighbor_list_accesses"] =
            metrics.l0_neighbor_list_accesses;
        result["l0_vector_accesses"] = metrics.l0_vector_accesses;
        result["neighbor_list_accesses"] =
            metrics.neighbor_list_accesses();
        result["vector_accesses"] = metrics.vector_accesses();
        return result;
    }

    size_t indexFileSize() const {
        return appr_alg->indexFileSize();
    }

    std::string dtypeName() const {
        return IndexDType<data_t>::name();
    }

    bool isFloatIndex() const {
        return std::is_same<data_t, float>::value;
    }

    size_t payloadBytesPerVector() const {
        return (size_t) dim * sizeof(data_t);
    }

    void validateHeaderLayout(const HnswHeaderInfo &header) const {
        size_t expected_payload = payloadBytesPerVector();
        size_t expected_offset_data = sizeof(hnswlib::linklistsizeint) + header.max_M0 * sizeof(hnswlib::tableint);

        if (header.offset_data != expected_offset_data)
            throw std::runtime_error("Typed HNSW metadata validation failed: offset_data does not match max_M0 layout.");
        if (header.label_offset < header.offset_data || header.label_offset - header.offset_data != expected_payload)
            throw std::runtime_error("Typed HNSW metadata validation failed: vector payload byte width does not match dtype and dim.");
        if (header.size_data_per_element != header.label_offset + sizeof(hnswlib::labeltype))
            throw std::runtime_error("Typed HNSW metadata validation failed: size_data_per_element does not match label offset.");
    }

    void writeTypedMetadata(const std::string &path_to_index) const {
        HnswHeaderInfo header = read_hnsw_header(path_to_index);
        validateHeaderLayout(header);

        std::ofstream output((path_to_index + ".hnswmeta.json").c_str());
        if (!output.is_open()) {
            throw std::runtime_error("Cannot create typed HNSW metadata sidecar.");
        }

        output << "{\n";
        output << "  \"format\": \"hnswlib-typed-index-metadata\",\n";
        output << "  \"version\": 1,\n";
        output << "  \"space\": \"" << json_escape(space_name) << "\",\n";
        output << "  \"dtype\": \"" << dtypeName() << "\",\n";
        output << "  \"dim\": " << dim << ",\n";
        output << "  \"distance_type\": \"float32\",\n";
        output << "  \"payload_bytes_per_vector\": " << payloadBytesPerVector() << ",\n";
        output << "  \"index_file\": \"" << json_escape(path_to_index) << "\",\n";
        output << "  \"index_file_size\": " << header.file_size << ",\n";
        output << "  \"max_elements\": " << header.max_elements << ",\n";
        output << "  \"cur_element_count\": " << header.cur_element_count << ",\n";
        output << "  \"size_data_per_element\": " << header.size_data_per_element << ",\n";
        output << "  \"label_offset\": " << header.label_offset << ",\n";
        output << "  \"offset_data\": " << header.offset_data << ",\n";
        output << "  \"max_M\": " << header.max_M << ",\n";
        output << "  \"max_M0\": " << header.max_M0 << ",\n";
        output << "  \"M\": " << header.M << ",\n";
        output << "  \"ef_construction\": " << header.ef_construction << "\n";
        output << "}\n";
    }

    void validateTypedMetadataForLoad(const std::string &path_to_index) const {
        std::string meta_path = path_to_index + ".hnswmeta.json";
        if (!file_exists(meta_path)) {
            if (isFloatIndex())
                return;
            throw std::runtime_error("Integer HNSW indexes require sidecar metadata: " + meta_path);
        }

        std::string json = read_text_file(meta_path);
        if (json_get_string(json, "format") != "hnswlib-typed-index-metadata")
            throw std::runtime_error("Typed HNSW metadata validation failed: unsupported format.");
        if (json_get_size_t(json, "version") != 1)
            throw std::runtime_error("Typed HNSW metadata validation failed: unsupported version.");
        if (json_get_string(json, "space") != space_name)
            throw std::runtime_error("Typed HNSW metadata validation failed: space mismatch.");
        if (json_get_string(json, "dtype") != dtypeName())
            throw std::runtime_error("Typed HNSW metadata validation failed: dtype mismatch.");
        if (json_get_size_t(json, "dim") != (size_t) dim)
            throw std::runtime_error("Typed HNSW metadata validation failed: dim mismatch.");
        if (json_get_string(json, "distance_type") != "float32")
            throw std::runtime_error("Typed HNSW metadata validation failed: distance_type mismatch.");
        if (json_get_size_t(json, "payload_bytes_per_vector") != payloadBytesPerVector())
            throw std::runtime_error("Typed HNSW metadata validation failed: payload byte width mismatch.");

        HnswHeaderInfo header = read_hnsw_header(path_to_index);
        validateHeaderLayout(header);

        if (json_get_size_t(json, "index_file_size") != header.file_size)
            throw std::runtime_error("Typed HNSW metadata validation failed: index file size mismatch.");
        if (json_get_size_t(json, "max_elements") != header.max_elements)
            throw std::runtime_error("Typed HNSW metadata validation failed: max_elements mismatch.");
        if (json_get_size_t(json, "cur_element_count") != header.cur_element_count)
            throw std::runtime_error("Typed HNSW metadata validation failed: cur_element_count mismatch.");
        if (json_get_size_t(json, "size_data_per_element") != header.size_data_per_element)
            throw std::runtime_error("Typed HNSW metadata validation failed: size_data_per_element mismatch.");
        if (json_get_size_t(json, "label_offset") != header.label_offset)
            throw std::runtime_error("Typed HNSW metadata validation failed: label_offset mismatch.");
        if (json_get_size_t(json, "offset_data") != header.offset_data)
            throw std::runtime_error("Typed HNSW metadata validation failed: offset_data mismatch.");
        if (json_get_size_t(json, "max_M") != header.max_M)
            throw std::runtime_error("Typed HNSW metadata validation failed: max_M mismatch.");
        if (json_get_size_t(json, "max_M0") != header.max_M0)
            throw std::runtime_error("Typed HNSW metadata validation failed: max_M0 mismatch.");
        if (json_get_size_t(json, "M") != header.M)
            throw std::runtime_error("Typed HNSW metadata validation failed: M mismatch.");
        if (json_get_size_t(json, "ef_construction") != header.ef_construction)
            throw std::runtime_error("Typed HNSW metadata validation failed: ef_construction mismatch.");
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
        if (!isFloatIndex()) {
            writeTypedMetadata(path_to_index);
        }
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
      validateTypedMetadataForLoad(path_to_index);
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }

    void loadIndexReadOnlyMmap(const std::string &path_to_index) {
      validateTypedMetadataForLoad(path_to_index);
      std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> loaded(
          new hnswlib::HierarchicalNSW<dist_t>(l2space));
      loaded->loadIndexReadOnlyMmap(path_to_index, l2space);
      delete appr_alg;
      appr_alg = loaded.release();
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false) {
        typename DataArrayCaster<data_t>::Array items = DataArrayCaster<data_t>::cast(input);
        auto buffer = items.request();
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                data_t* vector_data = (data_t*)items.data(0);
                std::vector<float> norm_array(dim);
                if (normalize) {
                    normalize_vector((float*)vector_data, norm_array.data());
                    appr_alg->addPoint((void*)norm_array.data(), (size_t)id, replace_deleted);
                } else {
                    appr_alg->addPoint((void*)vector_data, (size_t)id, replace_deleted);
                }
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)items.data(row), (size_t)id, replace_deleted);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)(norm_array.data() + start_idx), (size_t)id, replace_deleted);
                    });
            }
            cur_l += rows;
        }
    }


    py::object getData(py::object ids_ = py::none(), std::string return_type = "numpy") {
        std::vector<std::string> return_types{"numpy", "list"};
        if (std::find(std::begin(return_types), std::end(return_types), return_type) == std::end(return_types)) {
            throw std::invalid_argument("return_type should be \"numpy\" or \"list\"");
        }
        std::vector<size_t> ids;
        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();

            if (ids_numpy.ndim == 0) {
                throw std::invalid_argument("get_items accepts a list of indices and returns a list of vectors");
            } else {
                std::vector<size_t> ids1(ids_numpy.shape[0]);
                for (size_t i = 0; i < ids1.size(); i++) {
                    ids1[i] = items.data()[i];
                }
                ids.swap(ids1);
            }
        }

        std::vector<std::vector<data_t>> data;
        for (auto id : ids) {
            data.push_back(appr_alg->template getDataByLabel<data_t>(id));
        }
        if (return_type == "list") {
            return py::cast(data);
        }
        if (return_type == "numpy") {
            return py::array_t< data_t, py::array::c_style | py::array::forcecast >(py::cast(data));
        }
    }


    std::vector<hnswlib::labeltype> getIdsList() {
        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }


    py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        size_t level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        char* data_level0_npy = (char*)malloc(level0_npy_size);
        char* link_list_npy = (char*)malloc(link_npy_size);
        int* element_levels_npy = (int*)malloc(appr_alg->element_levels_.size() * sizeof(int));

        hnswlib::labeltype* label_lookup_key_npy = (hnswlib::labeltype*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        hnswlib::tableint* label_lookup_val_npy = (hnswlib::tableint*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        size_t idx = 0;
        for (auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it) {
            label_lookup_key_npy[idx] = it->first;
            label_lookup_val_npy[idx] = it->second;
            idx++;
        }

        memset(link_list_npy, 0, link_npy_size);

        memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
        memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize) {
                memcpy(link_list_npy + link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
            }
        }

        py::capsule free_when_done_l0(data_level0_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lvl(element_levels_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lb(label_lookup_key_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_id(label_lookup_val_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_ll(link_list_npy, [](void* f) {
            delete[] f;
            });

        /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
        /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */

        return py::dict(
            "offset_level0"_a = appr_alg->offsetLevel0_,
            "max_elements"_a = appr_alg->max_elements_,
            "cur_element_count"_a = (size_t)appr_alg->cur_element_count,
            "size_data_per_element"_a = appr_alg->size_data_per_element_,
            "label_offset"_a = appr_alg->label_offset_,
            "offset_data"_a = appr_alg->offsetData_,
            "max_level"_a = appr_alg->maxlevel_,
            "enterpoint_node"_a = appr_alg->enterpoint_node_,
            "max_M"_a = appr_alg->maxM_,
            "max_M0"_a = appr_alg->maxM0_,
            "M"_a = appr_alg->M_,
            "mult"_a = appr_alg->mult_,
            "ef_construction"_a = appr_alg->ef_construction_,
            "ef"_a = appr_alg->ef_,
            "has_deletions"_a = (bool)appr_alg->num_deleted_,
            "size_links_per_element"_a = appr_alg->size_links_per_element_,
            "allow_replace_deleted"_a = appr_alg->allow_replace_deleted_,

            "label_lookup_external"_a = py::array_t<hnswlib::labeltype>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                label_lookup_key_npy,  // the data pointer
                free_when_done_lb),

            "label_lookup_internal"_a = py::array_t<hnswlib::tableint>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::tableint) },  // C-style contiguous strides for each index
                label_lookup_val_npy,  // the data pointer
                free_when_done_id),

            "element_levels"_a = py::array_t<int>(
                { appr_alg->element_levels_.size() },  // shape
                { sizeof(int) },  // C-style contiguous strides for each index
                element_levels_npy,  // the data pointer
                free_when_done_lvl),

            // linkLists_,element_levels_,data_level0_memory_
            "data_level0"_a = py::array_t<char>(
                { level0_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                data_level0_npy,  // the data pointer
                free_when_done_l0),

            "link_lists"_a = py::array_t<char>(
                { link_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                link_list_npy,  // the data pointer
                free_when_done_ll));
    }


    py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        auto params = py::dict(
            "ser_version"_a = py::int_(TypedIndex<dist_t, data_t>::ser_version),  // serialization version
            "space"_a = space_name,
            "dtype"_a = dtypeName(),
            "dim"_a = dim,
            "index_inited"_a = index_inited,
            "ep_added"_a = ep_added,
            "normalize"_a = normalize,
            "num_threads"_a = num_threads_default,
            "seed"_a = seed);

        if (index_inited == false)
            return py::dict(**params, "ef"_a = default_ef);

        auto ann_params = getAnnData();

        return py::dict(**params, **ann_params);
    }


    static TypedIndex<dist_t, data_t>* createFromParams(const py::dict d) {
        // check serialization version
        assert_true(((int)py::int_(TypedIndex<dist_t, data_t>::ser_version)) >= d["ser_version"].cast<int>(), "Invalid serialization version!");

        auto space_name_ = d["space"].cast<std::string>();
        if (d.contains("dtype")) {
            assert_true(d["dtype"].cast<std::string>() == IndexDType<data_t>::name(), "Invalid dtype!");
        } else {
            assert_true(std::is_same<data_t, float>::value, "Missing dtype!");
        }
        auto dim_ = d["dim"].cast<int>();
        auto index_inited_ = d["index_inited"].cast<bool>();

        TypedIndex<dist_t, data_t>* new_index = new TypedIndex<dist_t, data_t>(space_name_, dim_);

        /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
        /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
        new_index->seed = d["seed"].cast<size_t>();

        if (index_inited_) {
            new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
                new_index->l2space,
                d["max_elements"].cast<size_t>(),
                d["M"].cast<size_t>(),
                d["ef_construction"].cast<size_t>(),
                new_index->seed);
            new_index->cur_l = d["cur_element_count"].cast<size_t>();
        }

        new_index->index_inited = index_inited_;
        new_index->ep_added = d["ep_added"].cast<bool>();
        new_index->num_threads_default = d["num_threads"].cast<int>();
        new_index->default_ef = d["ef"].cast<size_t>();

        if (index_inited_)
            new_index->setAnnData(d);

        return new_index;
    }


    static TypedIndex<dist_t, data_t> * createFromIndex(const TypedIndex<dist_t, data_t> & index) {
        return createFromParams(index.getIndexParams());
    }


    void setAnnData(const py::dict d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
        assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

        appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

        assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
        assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
        assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

        appr_alg->maxlevel_ = d["max_level"].cast<int>();
        appr_alg->enterpoint_node_ = d["enterpoint_node"].cast<hnswlib::tableint>();

        assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
        assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
        assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
        assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
        assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

        appr_alg->ef_ = d["ef"].cast<size_t>();

        assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

        auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t < hnswlib::labeltype, py::array::c_style | py::array::forcecast > >();
        auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t < hnswlib::tableint, py::array::c_style | py::array::forcecast > >();
        auto element_levels_npy = d["element_levels"].cast<py::array_t < int, py::array::c_style | py::array::forcecast > >();
        auto data_level0_npy = d["data_level0"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();
        auto link_list_npy = d["link_lists"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            if (label_lookup_val_npy.data()[i] < 0) {
                throw std::runtime_error("Internal id cannot be negative!");
            } else {
                appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
            }
        }

        memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

        for (size_t i = 0; i < appr_alg->max_elements_; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize == 0) {
                appr_alg->linkLists_[i] = nullptr;
            } else {
                appr_alg->linkLists_[i] = (char*)malloc(linkListSize);
                if (appr_alg->linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

                memcpy(appr_alg->linkLists_[i], link_list_npy.data() + link_npy_offsets[i], linkListSize);
            }
        }

        // process deleted elements
        bool allow_replace_deleted = false;
        if (d.contains("allow_replace_deleted")) {
            allow_replace_deleted = d["allow_replace_deleted"].cast<bool>();
        }
        appr_alg->allow_replace_deleted_= allow_replace_deleted;

        appr_alg->num_deleted_ = 0;
        bool has_deletions = d["has_deletions"].cast<bool>();
        if (has_deletions) {
            for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                if (appr_alg->isMarkedDeleted(i)) {
                    appr_alg->num_deleted_ += 1;
                    if (allow_replace_deleted) appr_alg->deleted_elements.insert(i);
                }
            }
        }
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        typename DataArrayCaster<data_t>::Array items = DataArrayCaster<data_t>::cast(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);
            if (features != dim)
                throw std::runtime_error("Wrong dimensionality of the vectors");

            // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            // Warning: search with a filter works slow in python in multithreaded mode. For best performance set num_threads=1
            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)items.data(row), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    float* data = (float*)items.data(row);

                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }

    py::object knnQueryWithLatency_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        typename DataArrayCaster<data_t>::Array items = DataArrayCaster<data_t>::cast(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        double* data_numpy_latency_ms;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);
            if (features != dim)
                throw std::runtime_error("Wrong dimensionality of the vectors");

            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];
            data_numpy_latency_ms = new double[rows];

            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    const auto start = std::chrono::steady_clock::now();
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)items.data(row), k, p_idFilter);
                    const auto finish = std::chrono::steady_clock::now();
                    data_numpy_latency_ms[row] =
                        std::chrono::duration<double, std::milli>(finish - start).count();
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    const auto start = std::chrono::steady_clock::now();
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    const auto finish = std::chrono::steady_clock::now();
                    data_numpy_latency_ms[row] =
                        std::chrono::duration<double, std::milli>(finish - start).count();
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_latency(data_numpy_latency_ms, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },
                data_numpy_l,
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },
                { k * sizeof(dist_t), sizeof(dist_t) },
                data_numpy_d,
                free_when_done_d),
            py::array_t<double>(
                { rows },
                { sizeof(double) },
                data_numpy_latency_ms,
                free_when_done_latency));
    }


    void markDeleted(size_t label) {
        appr_alg->markDelete(label);
    }


    void unmarkDeleted(size_t label) {
        appr_alg->unmarkDelete(label);
    }


    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }


    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }
};

class Index {
 public:
    std::string space_name;
    int dim;
    std::string dtype;

    std::unique_ptr<TypedIndex<float, float>> float_index;
    std::unique_ptr<TypedIndex<float, uint8_t>> uint8_index;
    std::unique_ptr<TypedIndex<float, int8_t>> int8_index;

    Index(const std::string &space_name, const int dim, const std::string &dtype = "float32")
        : space_name(space_name), dim(dim), dtype(dtype) {
        if (dtype == "float32") {
            float_index.reset(new TypedIndex<float, float>(space_name, dim));
        } else if (dtype == "uint8") {
            uint8_index.reset(new TypedIndex<float, uint8_t>(space_name, dim));
        } else if (dtype == "int8") {
            int8_index.reset(new TypedIndex<float, int8_t>(space_name, dim));
        } else {
            throw std::runtime_error("dtype must be one of float32, uint8, or int8.");
        }
    }

    void init_new_index(size_t maxElements, size_t M, size_t efConstruction, size_t random_seed, bool allow_replace_deleted) {
        if (float_index) return float_index->init_new_index(maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        if (uint8_index) return uint8_index->init_new_index(maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        return int8_index->init_new_index(maxElements, M, efConstruction, random_seed, allow_replace_deleted);
    }

    void set_ef(size_t ef) {
        if (float_index) return float_index->set_ef(ef);
        if (uint8_index) return uint8_index->set_ef(ef);
        return int8_index->set_ef(ef);
    }

    void set_num_threads(int num_threads) {
        if (float_index) return float_index->set_num_threads(num_threads);
        if (uint8_index) return uint8_index->set_num_threads(num_threads);
        return int8_index->set_num_threads(num_threads);
    }

    void setSearchAccessMetricsEnabled(bool enabled) {
        if (float_index) return float_index->setSearchAccessMetricsEnabled(enabled);
        if (uint8_index) return uint8_index->setSearchAccessMetricsEnabled(enabled);
        return int8_index->setSearchAccessMetricsEnabled(enabled);
    }

    void resetSearchAccessMetrics() {
        if (float_index) return float_index->resetSearchAccessMetrics();
        if (uint8_index) return uint8_index->resetSearchAccessMetrics();
        return int8_index->resetSearchAccessMetrics();
    }

    py::dict getSearchAccessMetrics() const {
        if (float_index) return float_index->getSearchAccessMetrics();
        if (uint8_index) return uint8_index->getSearchAccessMetrics();
        return int8_index->getSearchAccessMetrics();
    }

    int get_num_threads() const {
        if (float_index) return float_index->num_threads_default;
        if (uint8_index) return uint8_index->num_threads_default;
        return int8_index->num_threads_default;
    }

    size_t indexFileSize() const {
        if (float_index) return float_index->indexFileSize();
        if (uint8_index) return uint8_index->indexFileSize();
        return int8_index->indexFileSize();
    }

    void saveIndex(const std::string &path_to_index) {
        if (float_index) return float_index->saveIndex(path_to_index);
        if (uint8_index) return uint8_index->saveIndex(path_to_index);
        return int8_index->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
        if (float_index) return float_index->loadIndex(path_to_index, max_elements, allow_replace_deleted);
        if (uint8_index) return uint8_index->loadIndex(path_to_index, max_elements, allow_replace_deleted);
        return int8_index->loadIndex(path_to_index, max_elements, allow_replace_deleted);
    }

    void loadIndexReadOnlyMmap(const std::string &path_to_index) {
        if (float_index) return float_index->loadIndexReadOnlyMmap(path_to_index);
        if (uint8_index) return uint8_index->loadIndexReadOnlyMmap(path_to_index);
        return int8_index->loadIndexReadOnlyMmap(path_to_index);
    }

    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false) {
        if (float_index) return float_index->addItems(input, ids_, num_threads, replace_deleted);
        if (uint8_index) return uint8_index->addItems(input, ids_, num_threads, replace_deleted);
        return int8_index->addItems(input, ids_, num_threads, replace_deleted);
    }

    py::object getData(py::object ids_ = py::none(), std::string return_type = "numpy") {
        if (float_index) return float_index->getData(ids_, return_type);
        if (uint8_index) return uint8_index->getData(ids_, return_type);
        return int8_index->getData(ids_, return_type);
    }

    std::vector<hnswlib::labeltype> getIdsList() {
        if (float_index) return float_index->getIdsList();
        if (uint8_index) return uint8_index->getIdsList();
        return int8_index->getIdsList();
    }

    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        if (float_index) return float_index->knnQuery_return_numpy(input, k, num_threads, filter);
        if (uint8_index) return uint8_index->knnQuery_return_numpy(input, k, num_threads, filter);
        return int8_index->knnQuery_return_numpy(input, k, num_threads, filter);
    }

    py::object knnQueryWithLatency_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        if (float_index) return float_index->knnQueryWithLatency_return_numpy(input, k, num_threads, filter);
        if (uint8_index) return uint8_index->knnQueryWithLatency_return_numpy(input, k, num_threads, filter);
        return int8_index->knnQueryWithLatency_return_numpy(input, k, num_threads, filter);
    }

    void markDeleted(size_t label) {
        if (float_index) return float_index->markDeleted(label);
        if (uint8_index) return uint8_index->markDeleted(label);
        return int8_index->markDeleted(label);
    }

    void unmarkDeleted(size_t label) {
        if (float_index) return float_index->unmarkDeleted(label);
        if (uint8_index) return uint8_index->unmarkDeleted(label);
        return int8_index->unmarkDeleted(label);
    }

    void resizeIndex(size_t new_size) {
        if (float_index) return float_index->resizeIndex(new_size);
        if (uint8_index) return uint8_index->resizeIndex(new_size);
        return int8_index->resizeIndex(new_size);
    }

    size_t getMaxElements() const {
        if (float_index) return float_index->getMaxElements();
        if (uint8_index) return uint8_index->getMaxElements();
        return int8_index->getMaxElements();
    }

    size_t getCurrentCount() const {
        if (float_index) return float_index->getCurrentCount();
        if (uint8_index) return uint8_index->getCurrentCount();
        return int8_index->getCurrentCount();
    }

    py::dict getIndexParams() const {
        if (float_index) return float_index->getIndexParams();
        if (uint8_index) return uint8_index->getIndexParams();
        return int8_index->getIndexParams();
    }

    bool isInited() const {
        if (float_index) return float_index->index_inited;
        if (uint8_index) return uint8_index->index_inited;
        return int8_index->index_inited;
    }

    size_t getEf() const {
        if (float_index) return float_index->index_inited ? float_index->appr_alg->ef_ : float_index->default_ef;
        if (uint8_index) return uint8_index->index_inited ? uint8_index->appr_alg->ef_ : uint8_index->default_ef;
        return int8_index->index_inited ? int8_index->appr_alg->ef_ : int8_index->default_ef;
    }

    void setEf(size_t ef) {
        set_ef(ef);
    }

    size_t getMaxElementsProperty() const {
        if (float_index) return float_index->index_inited ? float_index->appr_alg->max_elements_ : 0;
        if (uint8_index) return uint8_index->index_inited ? uint8_index->appr_alg->max_elements_ : 0;
        return int8_index->index_inited ? int8_index->appr_alg->max_elements_ : 0;
    }

    size_t getElementCountProperty() const {
        if (float_index) return float_index->index_inited ? (size_t)float_index->appr_alg->cur_element_count : 0;
        if (uint8_index) return uint8_index->index_inited ? (size_t)uint8_index->appr_alg->cur_element_count : 0;
        return int8_index->index_inited ? (size_t)int8_index->appr_alg->cur_element_count : 0;
    }

    size_t getEfConstructionProperty() const {
        if (float_index) return float_index->index_inited ? float_index->appr_alg->ef_construction_ : 0;
        if (uint8_index) return uint8_index->index_inited ? uint8_index->appr_alg->ef_construction_ : 0;
        return int8_index->index_inited ? int8_index->appr_alg->ef_construction_ : 0;
    }

    size_t getMProperty() const {
        if (float_index) return float_index->index_inited ? float_index->appr_alg->M_ : 0;
        if (uint8_index) return uint8_index->index_inited ? uint8_index->appr_alg->M_ : 0;
        return int8_index->index_inited ? int8_index->appr_alg->M_ : 0;
    }

    static Index* createFromParams(const py::dict d) {
        std::string dtype_ = d.contains("dtype") ? d["dtype"].cast<std::string>() : "float32";
        Index *index = new Index(d["space"].cast<std::string>(), d["dim"].cast<int>(), dtype_);
        if (dtype_ == "float32") {
            index->float_index.reset(TypedIndex<float, float>::createFromParams(d));
        } else if (dtype_ == "uint8") {
            index->uint8_index.reset(TypedIndex<float, uint8_t>::createFromParams(d));
        } else if (dtype_ == "int8") {
            index->int8_index.reset(TypedIndex<float, int8_t>::createFromParams(d));
        } else {
            delete index;
            throw std::runtime_error("dtype must be one of float32, uint8, or int8.");
        }
        return index;
    }

    static Index* createFromIndex(const Index &index) {
        return createFromParams(index.getIndexParams());
    }
};

template<typename dist_t, typename data_t = float>
class BFIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    bool index_inited;
    bool normalize;
    int num_threads_default;

    hnswlib::labeltype cur_l;
    hnswlib::BruteforceSearch<dist_t>* alg;
    hnswlib::SpaceInterface<float>* space;


    BFIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        alg = NULL;
        index_inited = false;

        num_threads_default = std::thread::hardware_concurrency();
    }


    ~BFIndex() {
        delete space;
        if (alg)
            delete alg;
    }


    size_t getMaxElements() const {
        return alg->maxelements_;
    }


    size_t getCurrentCount() const {
        return alg->cur_element_count;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }


    void init_new_index(const size_t maxElements) {
        if (alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        alg = new hnswlib::BruteforceSearch<dist_t>(space, maxElements);
        index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none()) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            for (size_t row = 0; row < rows; row++) {
                size_t id = ids.size() ? ids.at(row) : cur_l + row;
                if (!normalize) {
                    alg->addPoint((void *) items.data(row), (size_t) id);
                } else {
                    std::vector<float> normalized_vector(dim);
                    normalize_vector((float *)items.data(row), normalized_vector.data());
                    alg->addPoint((void *) normalized_vector.data(), (size_t) id);
                }
            }
            cur_l+=rows;
        }
    }


    void deleteVector(size_t label) {
        alg->removePoint(label);
    }


    void saveIndex(const std::string &path_to_index) {
        alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
            delete alg;
        }
        alg = new hnswlib::BruteforceSearch<dist_t>(space, path_to_index);
        cur_l = alg->cur_element_count;
        index_inited = true;
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype *data_numpy_l;
        dist_t *data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            if (!normalize) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
                        (void*)items.data(row), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. There are not enough elements.");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), norm_array.data() + start_idx);

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. There are not enough elements.");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }

        py::capsule free_when_done_l(data_numpy_l, [](void *f) {
            delete[] f;
        });
        py::capsule free_when_done_d(data_numpy_d, [](void *f) {
            delete[] f;
        });


        return py::make_tuple(
                py::array_t<hnswlib::labeltype>(
                        { rows, k },  // shape
                        { k * sizeof(hnswlib::labeltype),
                          sizeof(hnswlib::labeltype)},  // C-style contiguous strides for each index
                        data_numpy_l,  // the data pointer
                        free_when_done_l),
                py::array_t<dist_t>(
                        { rows, k },  // shape
                        { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                        data_numpy_d,  // the data pointer
                        free_when_done_d));
    }
};


PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");

        py::class_<Index>(m, "Index")
        .def(py::init(&Index::createFromParams), py::arg("params"))
           /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
        .def(py::init(&Index::createFromIndex), py::arg("index"))
        .def(py::init<const std::string &, const int, const std::string &>(), py::arg("space"), py::arg("dim"), py::arg("dtype") = "float32")
        .def("init_index",
            &Index::init_new_index,
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false)
        .def("knn_query",
            &Index::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("knn_query_with_latency",
            &Index::knnQueryWithLatency_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items",
            &Index::addItems,
            py::arg("data"),
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false)
        .def("get_items", &Index::getData, py::arg("ids") = py::none(), py::arg("return_type") = "numpy")
        .def("get_ids_list", &Index::getIdsList)
        .def("set_ef", &Index::set_ef, py::arg("ef"))
        .def("set_num_threads", &Index::set_num_threads, py::arg("num_threads"))
        .def("set_search_access_metrics_enabled",
            &Index::setSearchAccessMetricsEnabled,
            py::arg("enabled"))
        .def("reset_search_access_metrics",
            &Index::resetSearchAccessMetrics)
        .def("get_search_access_metrics",
            &Index::getSearchAccessMetrics)
        .def("index_file_size", &Index::indexFileSize)
        .def("save_index", &Index::saveIndex, py::arg("path_to_index"))
        .def("load_index",
            &Index::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("allow_replace_deleted") = false)
        .def("load_index_readonly_mmap",
            &Index::loadIndexReadOnlyMmap,
            py::arg("path_to_index"))
        .def("mark_deleted", &Index::markDeleted, py::arg("label"))
        .def("unmark_deleted", &Index::unmarkDeleted, py::arg("label"))
        .def("resize_index", &Index::resizeIndex, py::arg("new_size"))
        .def("get_max_elements", &Index::getMaxElements)
        .def("get_current_count", &Index::getCurrentCount)
        .def_readonly("space", &Index::space_name)
        .def_readonly("dim", &Index::dim)
        .def_readonly("dtype", &Index::dtype)
        .def_property("num_threads", &Index::get_num_threads, &Index::set_num_threads)
        .def_property("ef",
          &Index::getEf,
          &Index::setEf)
        .def_property_readonly("max_elements", &Index::getMaxElementsProperty)
        .def_property_readonly("element_count", &Index::getElementCountProperty)
        .def_property_readonly("ef_construction", &Index::getEfConstructionProperty)
        .def_property_readonly("M", &Index::getMProperty)

        .def(py::pickle(
            [](const Index &ind) {  // __getstate__
                return py::make_tuple(ind.getIndexParams()); /* Return dict (wrapped in a tuple) that fully encodes state of the Index object */
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return Index::createFromParams(t[0].cast<py::dict>());
            }))

        .def("__repr__", [](const Index &a) {
            return "<hnswlib.Index(space='" + a.space_name + "', dim="+std::to_string(a.dim)+", dtype='"+a.dtype+"')>";
        });

        py::class_<BFIndex<float>>(m, "BFIndex")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &BFIndex<float>::init_new_index, py::arg("max_elements"))
        .def("knn_query",
            &BFIndex<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items", &BFIndex<float>::addItems, py::arg("data"), py::arg("ids") = py::none())
        .def("delete_vector", &BFIndex<float>::deleteVector, py::arg("label"))
        .def("set_num_threads", &BFIndex<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &BFIndex<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index", &BFIndex<float>::loadIndex, py::arg("path_to_index"), py::arg("max_elements") = 0)
        .def("__repr__", [](const BFIndex<float> &a) {
            return "<hnswlib.BFIndex(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        })
        .def("get_max_elements", &BFIndex<float>::getMaxElements)
        .def("get_current_count", &BFIndex<float>::getCurrentCount)
        .def_readwrite("num_threads", &BFIndex<float>::num_threads_default);
        return m.ptr();
}
