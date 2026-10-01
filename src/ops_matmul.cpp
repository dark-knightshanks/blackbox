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
    const float* in_ptr = reinterpret_cast<const float*>(input.raw_data());
    const float* weight_ptr = reinterpret_cast<const float*>(weights.raw_data());
    
    const float* bias_ptr = nullptr;
    if (!bias.shape.empty() && bias.raw_data() != nullptr) {
        bias_ptr = reinterpret_cast<const float*>(bias.raw_data());
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
            if (bias_ptr != nullptr && j >= 0 && j < static_cast<int>(bias.size())) {
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
    std::cout << "Duration: " << duration << " us\n";

    return output;
}

Tensor run_matmul(const Tensor& A, const Tensor& B){

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    int64_t batch = A.shape[0];
    int64_t heads = A.shape[1];
    int64_t M = A.shape[2];
    int64_t K = A.shape[3];
    int64_t N = B.shape[3];
    if(A.shape[3] != B.shape[2]){
        throw std::runtime_error("Invalid matrix shape for MatMul: A.shape[3] != B.shape[2]");
    } 

    output.shape = {batch, heads, M, N};
    output.data.resize(output.byte_size());
    const float* A_ptr = reinterpret_cast<const float*>(A.raw_data());
    const float* B_ptr = reinterpret_cast<const float*>(B.raw_data());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
                                          
    for(int64_t b=0; b<batch; ++b){
        for(int64_t h=0; h<heads; ++h){
            int64_t b_a = (A.shape[0] == 1) ? 0 : b;                                 
            int64_t b_b = (B.shape[0] == 1) ? 0 : b;                                                                                                  
            int64_t h_a = (A.shape[1] == 1) ? 0 : h;                                 
            int64_t h_b = (B.shape[1] == 1) ? 0 : h;
            int64_t offset_A = (b_a*A.shape[1] + h_a)*(M*K);
            int64_t offset_B = (b_b*B.shape[1] + h_b)*(K*N);
            int64_t offset_Out = (b * heads + h) * (M * N);
            for(int64_t m=0; m<M; ++m){
                for(int64_t n=0; n<N; ++n){
                    float sum = 0.0f;
                    for(int64_t k=0; k<K; ++k){
                        float a_val = A_ptr[offset_A + (m*K + k)];
                        float b_val = B_ptr[offset_B + (k*N + n)];
                        sum += a_val*b_val;
                    }
                    out_ptr[offset_Out + (m*N + n)] = sum;  
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";

    return output;
}
