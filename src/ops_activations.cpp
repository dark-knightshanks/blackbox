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

Tensor run_GELU(const Tensor& input){
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());

    double a = sqrt(2.0f / PI);
    for(size_t i = 0; i<input.size();++i){
        float x = in_ptr[i];
        out_ptr[i] = 0.5 * x * (1 + tanh(a * (x + (0.044715*(x*x*x)))));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";
    return output;
}

Tensor run_SilU(const Tensor& input){
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());

    for(size_t i = 0; i<input.size();++i){
        float x = in_ptr[i];
        float sigmoid = 1.0f /(1.0f + (exp(-1*x)));
        out_ptr[i] = x * sigmoid;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";
    return output;
}

Tensor run_softmax(const Tensor& input, int64_t axis){
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    int rank = input.shape.size();                                     
    int64_t ax = axis;
    if (ax < 0){                                                        
        ax += rank;
    }
    int64_t dim = input.shape[ax];
    int outer_size = input.size()/dim;
    for(int i=0; i<outer_size; ++i){
        int64_t offset = i * dim; 
        float max_val = in_ptr[offset];
        for (int k = 1; k < dim; ++k) {
            if (in_ptr[offset + k] > max_val) {
                max_val = in_ptr[offset + k];
            }
        }
        float sum = 0.0f;
        for (int k = 0; k < dim; ++k) {
            float exp_val = exp(in_ptr[offset + k] - max_val);
            out_ptr[offset + k] = exp_val;
            sum += exp_val;
        }
        float inv_sum = 1.0f / sum;
        for(int k = 0; k < dim; ++k){
            out_ptr[offset + k] *= inv_sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout<<"Duration :"<<duration.count()<<" us\n";

    return output;
}
