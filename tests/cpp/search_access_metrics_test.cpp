#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <atomic>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

namespace {

std::atomic<uint64_t> distance_calls{0};

float counting_l2(const void *left, const void *right, const void *dimension_ptr) {
    distance_calls.fetch_add(1, std::memory_order_relaxed);
    const size_t dimension = *static_cast<const size_t *>(dimension_ptr);
    const float *lhs = static_cast<const float *>(left);
    const float *rhs = static_cast<const float *>(right);
    float distance = 0.0f;
    for (size_t index = 0; index < dimension; ++index) {
        const float delta = lhs[index] - rhs[index];
        distance += delta * delta;
    }
    return distance;
}

class CountingL2Space : public hnswlib::SpaceInterface<float> {
 public:
    explicit CountingL2Space(size_t dimension)
        : dimension_(dimension), data_size_(dimension * sizeof(float)) {}

    size_t get_data_size() override {
        return data_size_;
    }

    hnswlib::DISTFUNC<float> get_dist_func() override {
        return counting_l2;
    }

    void *get_dist_func_param() override {
        return &dimension_;
    }

 private:
    size_t dimension_;
    size_t data_size_;
};

using SearchResult =
    std::priority_queue<std::pair<float, hnswlib::labeltype>>;

std::vector<std::pair<float, hnswlib::labeltype>>
consume(SearchResult result) {
    std::vector<std::pair<float, hnswlib::labeltype>> values;
    while (!result.empty()) {
        values.push_back(result.top());
        result.pop();
    }
    return values;
}

void test_search_access_metrics() {
    constexpr size_t dimension = 4;
    constexpr size_t point_count = 512;
    constexpr size_t k = 10;
    CountingL2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(
        &space, point_count, 16, 200, 100);

    std::vector<float> points(point_count * dimension);
    for (size_t point = 0; point < point_count; ++point) {
        for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
            points[point * dimension + coordinate] =
                static_cast<float>((point * 17 + coordinate * 13) % 101) /
                101.0f;
        }
        index.addPoint(points.data() + point * dimension, point);
    }
    index.setEf(64);

    const float query[dimension] = {0.15f, 0.35f, 0.55f, 0.75f};
    index.resetSearchAccessMetrics();
    distance_calls.store(0, std::memory_order_relaxed);
    const auto first_result = consume(index.searchKnn(query, k));
    const hnswlib::SearchAccessMetrics first =
        index.getSearchAccessMetrics();

    assert(first.entrypoint_vector_accesses == 1);
    assert(first.upper_neighbor_list_accesses > 0);
    assert(first.upper_vector_accesses > 0);
    assert(first.l0_neighbor_list_accesses > 0);
    assert(first.l0_vector_accesses > 0);
    assert(first.vector_accesses() ==
           distance_calls.load(std::memory_order_relaxed));
    assert(first.neighbor_list_accesses() ==
           first.upper_neighbor_list_accesses +
               first.l0_neighbor_list_accesses);

    index.resetSearchAccessMetrics();
    const hnswlib::SearchAccessMetrics reset =
        index.getSearchAccessMetrics();
    assert(reset.entrypoint_vector_accesses == 0);
    assert(reset.upper_neighbor_list_accesses == 0);
    assert(reset.upper_vector_accesses == 0);
    assert(reset.l0_neighbor_list_accesses == 0);
    assert(reset.l0_vector_accesses == 0);

    distance_calls.store(0, std::memory_order_relaxed);
    const auto second_result = consume(index.searchKnn(query, k));
    const hnswlib::SearchAccessMetrics second =
        index.getSearchAccessMetrics();
    assert(first_result == second_result);
    assert(second.vector_accesses() ==
           distance_calls.load(std::memory_order_relaxed));
}

}  // namespace

int main() {
    std::cout << "Testing search access metrics ..." << std::endl;
    test_search_access_metrics();
    std::cout << "Test ok" << std::endl;
    return 0;
}
