#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>
#include <numeric>

Tensor run_layernorm(Tensor& input, const Tensor& weights, const Tensor& bias){

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    int batch, length, dims;
    batch = input.shape[0];
    length = input.shape[1];// patches for ViT, tokens for LLM's
    dims = input.shape[2];
    const float* in_ptr = reinterpret_cast<const float*>(input.raw_data());
    const float* weight_ptr = reinterpret_cast<const float*>(weights.raw_data());
    const float* bias_ptr = nullptr;
    if (!bias.shape.empty() && bias.raw_data() != nullptr) {
        bias_ptr = reinterpret_cast<const float*>(bias.raw_data());
    }
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for(int i=0; i<batch; ++i){
        for(int j=0; j<length; ++j){
            int64_t offset = (i * length + j) * dims; 
            float sum = 0.0f, sum_var = 0.0f, diff = 0.0f, var = 0.0f, mean = 0.0f;  
            for(int k=0; k<dims; ++k){
                sum += in_ptr[offset + k];     
            }
            mean = sum/dims;
            for(int k=0; k<dims; ++k){
                diff = in_ptr[offset + k] - mean;
                sum_var += diff*diff;
            }
            var = sum_var/dims;
            for(int k=0; k<dims; ++k){
                float norm = (in_ptr[offset + k] - mean)/(sqrt(var + 1e-5f));                      
                out_ptr[offset + k] = norm * weight_ptr[k] + (bias_ptr ? bias_ptr[k] : 0.0f);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";

    return output;
}

Tensor run_rmsnorm(const Tensor& input, const Tensor& weights){

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    int batch, length, dims;
    batch = input.shape[0];
    length = input.shape[1];// patches for ViT, tokens for LLM's
    dims = input.shape[2];
    const float* in_ptr = reinterpret_cast<const float*>(input.raw_data());
    const float* weight_ptr = reinterpret_cast<const float*>(weights.raw_data());
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for(int i=0; i<batch; ++i){
        for(int j=0; j<length; ++j){
            float sum, mean, rms;
            int64_t offset = (i * length + j) * dims; 

            sum = 0.0f;
            for(int k=0; k<dims; ++k){
                sum += (in_ptr[offset + k]*in_ptr[offset + k]);     
            }
            mean = sum/dims;
            rms = sqrt(mean + 1e-5f);
            for(int k=0; k<dims; ++k){
                out_ptr[offset + k] = weight_ptr[k]*((in_ptr[offset + k])/(rms));
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";

    return output;
}