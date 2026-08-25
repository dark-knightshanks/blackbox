#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>
#include <numeric>

int argmax(const Tensor& output){
    int max_idx = 0;
    const float* out_ptr = reinterpret_cast<const float*>(output.data.data());
    float max_value = out_ptr[0];

    for(size_t i=0; i<output.size(); ++i){
        if(out_ptr[i]>max_value){
            max_value = out_ptr[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

static std::vector<int64_t> parse_target_shape(const Tensor& shape) {
    std::vector<int64_t> target_dims;
    if (shape.data.empty()) return target_dims;

    size_t expected_dims = shape.shape.empty() ? (shape.data.size() / 2) : shape.shape[0];
    if (expected_dims == 0) expected_dims = shape.data.size();

    // Check if data is stored as raw 64-bit integers
    if (shape.data.size() == expected_dims * sizeof(int64_t)) {
        const int64_t* ptr = reinterpret_cast<const int64_t*>(shape.data.data());
        for (size_t i = 0; i < expected_dims; ++i) {
            target_dims.push_back(ptr[i]);
        }
    } else {
        for (size_t i = 0; i < expected_dims; ++i) {
            target_dims.push_back(static_cast<int64_t>(shape.data[i]));
        }
    }
    return target_dims;
}

Tensor run_reshape(const Tensor& input, const Tensor& shape) {
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    output.data = input.data; // Directly copies the raw data

    std::vector<int64_t> target_dims = parse_target_shape(shape);
    int64_t total_elements = input.size();
    int64_t known_product = 1;
    int minus_one_index = -1;

    for (size_t i = 0; i < target_dims.size(); ++i) {
        int64_t target_dim = target_dims[i];

        if (target_dim == 0) {
            int64_t dim = (i < input.shape.size()) ? input.shape[i] : 1;
            output.shape.push_back(dim);
            known_product *= dim;
        } 
        else if (target_dim == -1) {
            minus_one_index = static_cast<int>(i);
            output.shape.push_back(-1);
        } 
        else {
            output.shape.push_back(target_dim);
            known_product *= target_dim;
        }
    }

    // Solve for -1 if it was present
    if (minus_one_index != -1) {
        output.shape[minus_one_index] = total_elements / known_product;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout<<"Duration :"<<duration.count()<<" us\n";

    return output;
}

