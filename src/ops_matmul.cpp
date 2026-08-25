#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>
#include <numeric>

Tensor run_gemm(const Tensor& input, const Tensor& weights, const Tensor& bias, int transB) {
    
    auto start = std::chrono::high_resolution_clock::now();
    if (input.shape.empty() || weights.shape.size() < 2) {
        throw std::runtime_error("run_gemm: input or weight tensor has invalid shape!");
    }

    Tensor output;
    int K, N;
   if (transB == 1) {
        N = weights.shape[0]; // Out features (10)
        K = weights.shape[1]; // In features (256)
    } else {
        K = weights.shape[0]; // In features
        N = weights.shape[1]; // Out features
    }

    // Safely calculate batch size M based on total input floats vs K
int M = (K > 0) ? static_cast<int>(input.size() / K) : 1;
    if (M <= 0) M = 1;

    output.shape = {M, N};
    output.data.resize(output.byte_size());
    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    const float* weight_ptr = reinterpret_cast<const float*>(weights.data.data());
    
    const float* bias_ptr = nullptr;
    if (!bias.data.empty()) {
        bias_ptr = reinterpret_cast<const float*>(bias.data.data());
    }
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int i1 = (i * K) + k;
                int i2 = (transB == 1) ? (j * K + k) : (k * N + j);

                float in_val = (i1 >= 0 && i1 < static_cast<int>(input.size())) ? in_ptr[i1] : 0.0f;
                float w_val = (i2 >= 0 && i2 < static_cast<int>(weights.size())) ? weight_ptr[i2] : 0.0f;

                sum += (in_val * w_val);    
            }

            // Safe bias addition
            if (!bias.data.empty() && j >= 0 && j < static_cast<int>(bias.size())) {
                sum += bias_ptr[j];
            }
            int out_index = (i * N) + j;
            if (out_index >= 0 && out_index < static_cast<int>(output.size())) {
                out_ptr[out_index] = sum;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";

    return output;
}
