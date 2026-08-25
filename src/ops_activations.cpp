#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>
#include <numeric>

Tensor run_relu(const Tensor& input){
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for(size_t i = 0; i<input.size();++i){
        out_ptr[i]=std::max(0.0f, in_ptr[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";
    return output;
}

