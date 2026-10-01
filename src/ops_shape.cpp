#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>
#include <numeric>

int argmax(const Tensor& output){
    int max_idx = 0;
    const float* out_ptr = reinterpret_cast<const float*>(output.raw_data());
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
    const uint8_t* raw = shape.raw_data();
    if (raw == nullptr) return target_dims;

    size_t count = shape.size();
    if (count == 0) {
        count = shape.data.size() / sizeof(int64_t);
    }
    if (count == 0) return target_dims;

    const int64_t* ptr = reinterpret_cast<const int64_t*>(raw);
    for (size_t i = 0; i < count; ++i) {
        target_dims.push_back(ptr[i]);
    }
    return target_dims;
}

Tensor run_reshape(const Tensor& input, const Tensor& shape) {
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    output.external_ptr = input.external_ptr;
    if (output.external_ptr == nullptr) {
        output.data = input.data; // Directly copies the raw data
    }

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

Tensor run_transpose(const Tensor& input,  const std::vector<int64_t>& perm){
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    int64_t A,B,C,D,p0,p1,p2,p3;
    p0 = perm[0];
    p1 = perm[1]; 
    p2 = perm[2];
    p3 = perm[3];
    A = input.shape[0];
    B = input.shape[1];
    C = input.shape[2];
    D = input.shape[3];
    output.shape = {input.shape[p0], input.shape[p1], input.shape[p2], input.shape[p3]};
    output.data.resize(output.byte_size());
    const float* in_ptr = reinterpret_cast<const float*>(input.raw_data());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for(int i=0; i<A; ++i){
        for(int j=0; j<B; ++j){
            for(int k=0; k<C; ++k){
                for(int l=0; l<D; ++l){
                    int64_t in_idx = (i*B*C*D) + (j*C*D) + (k*D) + l;
                    int64_t cord[4] = {i,j,k,l};
                    int64_t out_i = cord[p0];
                    int64_t out_j = cord[p1];
                    int64_t out_k = cord[p2];
                    int64_t out_l = cord[p3];
                    int64_t out_dim1 = output.shape[1];
                    int64_t out_dim2 = output.shape[2];
                    int64_t out_dim3 = output.shape[3];
                    int64_t out_idx = (out_i*out_dim1 + out_j)*(out_dim2)*(out_dim3) + (out_k*out_dim3) + out_l;
                    out_ptr[out_idx] = in_ptr[in_idx];
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout<<"Duration :"<<duration.count()<<" us\n";

    return output;
}

Tensor run_add(const Tensor& a, const Tensor& b){
    auto start = std::chrono::high_resolution_clock::now();
    const Tensor& big = (a.size() >= b.size()) ? a : b;
    const Tensor& small = (a.size() >= b.size()) ? b : a;
    Tensor output;
    output.shape = big.shape;
    output.data.resize(big.byte_size());
    const float* big_ptr = reinterpret_cast<const float*>(big.data.data());
    const float* small_ptr = reinterpret_cast<const float*>(small.data.data());
    float* out_ptr = reinterpret_cast<float *>(output.data.data());
    for(size_t i = 0 ; i < big.size() ; ++i){
        out_ptr[i] = big_ptr[i] + small_ptr[i%small.size()];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout<<"Duration :"<<duration.count()<<" us\n";
    return output;

}