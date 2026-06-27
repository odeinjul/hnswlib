#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

template <typename T>
float reference_l2(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        result += diff * diff;
    }
    return result;
}

void test_uint8_distance() {
    std::vector<size_t> dims{1, 2, 3, 4, 15, 16, 17, 31, 32, 64, 100, 128};
    for (size_t dim : dims) {
        std::vector<uint8_t> a(dim);
        std::vector<uint8_t> b(dim);
        for (size_t i = 0; i < dim; i++) {
            a[i] = static_cast<uint8_t>((i * 37) & 0xff);
            b[i] = static_cast<uint8_t>(255 - ((i * 19) & 0xff));
        }
        if (dim >= 4) {
            a[0] = 0;
            a[1] = 1;
            a[2] = 254;
            a[3] = 255;
        }

        hnswlib::L2SpaceUInt8 space(dim);
        float expected = reference_l2(a, b);
        float scalar = hnswlib::L2SqrUInt8(a.data(), b.data(), &dim);
        float selected = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        assert(std::fabs(scalar - expected) < 1e-5f);
        assert(std::fabs(selected - expected) < 1e-5f);
        assert(space.get_data_size() == dim * sizeof(uint8_t));
    }
}

void test_int8_distance() {
    std::vector<size_t> dims{1, 2, 3, 4, 15, 16, 17, 31, 32, 64, 100, 128};
    for (size_t dim : dims) {
        std::vector<int8_t> a(dim);
        std::vector<int8_t> b(dim);
        for (size_t i = 0; i < dim; i++) {
            a[i] = static_cast<int8_t>((static_cast<int>(i * 11) % 255) - 128);
            b[i] = static_cast<int8_t>(127 - (static_cast<int>(i * 7) % 255));
        }
        if (dim >= 5) {
            a[0] = -128;
            a[1] = -127;
            a[2] = 0;
            a[3] = 126;
            a[4] = 127;
        }

        hnswlib::L2SpaceInt8 space(dim);
        float expected = reference_l2(a, b);
        float scalar = hnswlib::L2SqrInt8(a.data(), b.data(), &dim);
        float selected = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        assert(std::fabs(scalar - expected) < 1e-5f);
        assert(std::fabs(selected - expected) < 1e-5f);
        assert(space.get_data_size() == dim * sizeof(int8_t));
    }
}

void test_uint8_hnsw_save_load() {
    const size_t dim = 8;
    const size_t count = 16;
    hnswlib::L2SpaceUInt8 space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, count, 8, 40);

    std::vector<std::vector<uint8_t>> data(count, std::vector<uint8_t>(dim));
    for (size_t row = 0; row < count; row++) {
        for (size_t col = 0; col < dim; col++) {
            data[row][col] = static_cast<uint8_t>(row * 10 + col);
        }
        index.addPoint(data[row].data(), row);
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = index.searchKnn(data[3].data(), 1);
    assert(result.top().second == 3);

    const std::string path = "/tmp/hnswlib_uint8_cpp_test.bin";
    index.saveIndex(path);

    hnswlib::HierarchicalNSW<float> loaded(&space, path);
    std::priority_queue<std::pair<float, hnswlib::labeltype>> loaded_result = loaded.searchKnn(data[7].data(), 1);
    assert(loaded_result.top().second == 7);

    assert(loaded.label_offset_ - loaded.offsetData_ == dim);
    assert(loaded.size_data_per_element_ == loaded.label_offset_ + sizeof(hnswlib::labeltype));
    std::remove(path.c_str());
}

int main() {
    test_uint8_distance();
    test_int8_distance();
    test_uint8_hnsw_save_load();
    return 0;
}
