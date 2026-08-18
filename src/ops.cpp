#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>


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

Tensor run_maxpool2D(const Tensor& input,const std::vector<int64_t>& kernel, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads){
    auto start = std::chrono::high_resolution_clock::now();
    if (input.shape.size() < 4) {
        throw std::runtime_error("run_maxpool2D: input tensor must be 4D!");
    }
    Tensor output;
    int64_t N = input.shape[0];
    int64_t C = input.shape[1];
    int64_t H = input.shape[2];
    int64_t W = input.shape[3];
    int64_t Kh = kernel[0];
    int64_t Kw = kernel[1];
    int64_t Sh = strides.empty() ? 1 : strides[0];
    int64_t Sw = (strides.size() >= 2) ? strides[1] : Sh;
    if (Sh <= 0) Sh = 1;
    if (Sw <= 0) Sw = 1;
    int64_t Ph = pads.empty() ? 0 : pads[0];
    int64_t Pw = pads.size() < 2 ? Ph : pads[1];

    int64_t H_out = (H + 2 * Ph - Kh) / Sh + 1;
    int64_t W_out = (W + 2 * Pw - Kw) / Sw + 1;
    output.shape = {N, C, H_out, W_out};
    output.data.resize(output.byte_size());
    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());

    for(int64_t n=0; n<N; ++n){
        for(int64_t c=0; c<C; ++c){
            for(int64_t h=0; h<H_out; ++h){
                for(int64_t w=0; w<W_out; ++w){
                    float max_val = -1e9f;
                    for(int64_t kh=0; kh<Kh; ++kh){
                        for(int64_t kw=0; kw<Kw; ++kw){
                            // Map to input coordinates
                            int64_t ih = (h * Sh) + kh - Ph;
                            int64_t iw = (w * Sw) + kw - Pw;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {           
                                int64_t in_idx = (n * C * H * W) + (c * H * W) + (ih * W) + iw;
                                if (in_idx < static_cast<int64_t>(input.size())) {
                                    max_val = std::max(max_val, in_ptr[in_idx]);
                                }
                            }
                        }
                    }
                    // Save max result to output tensor
                    int64_t out_idx = (n * C * H_out * W_out) + (c * H_out * W_out) + (h * W_out) + w;
                    if (out_idx < static_cast<int64_t>(output.size())) {
                        out_ptr[out_idx] = max_val;
                    }
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";
    return output;
}

Tensor run_conv2D(const Tensor& input, const Tensor& weights, 
                    const Tensor& bias, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads){
    auto start = std::chrono::high_resolution_clock::now();
if (input.shape.size() < 4) {
        throw std::runtime_error("run_conv2D: input tensor is not 4D! Actual size: " + std::to_string(input.shape.size()));
    }
    if (weights.shape.size() < 4) {
        throw std::runtime_error("run_conv2D: weights tensor is not 4D! Actual size: " + std::to_string(weights.shape.size()));
    }

    Tensor output;
    int64_t N = input.shape[0];
    int64_t C = input.shape[1];
    int64_t H = input.shape[2];
    int64_t W = input.shape[3];
    int64_t C_out = weights.shape[0];
    int64_t Kh = weights.shape[2];
    int64_t Kw = weights.shape[3];
    // Bounds-safe stride extraction
    int64_t Sh = strides.empty() ? 1 : strides[0];
    int64_t Sw = (strides.size() >= 2) ? strides[1] : Sh;
    if (Sh <= 0) Sh = 1;
    if (Sw <= 0) Sw = 1;

    // Bounds-safe padding extraction
    int64_t Ph = pads.empty() ? 0 : pads[0];
    int64_t Pw = (pads.size() >= 2) ? pads[1] : Ph;

    int64_t H_out = (H + 2 * Ph - Kh) / Sh + 1;
    int64_t W_out = (W + 2 * Pw - Kw) / Sw + 1;

    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("run_conv2D error: invalid output dimensions calculated: " + 
                                 std::to_string(H_out) + "x" + std::to_string(W_out));
    }

    output.shape = {N, C_out, H_out, W_out};
    output.data.resize(output.byte_size());
    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    const float* weight_ptr = reinterpret_cast<const float*>(weights.data.data());
    
    const float* bias_ptr = nullptr;
    if (!bias.data.empty()) {
        bias_ptr = reinterpret_cast<const float*>(bias.data.data());
    }
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for(int64_t n=0; n<N; ++n){
        for(int64_t oc=0; oc<C_out; ++oc){
            for(int64_t h=0; h<H_out; ++h){
                for(int64_t w=0; w<W_out; ++w){
                    float sum = 0.0f;
                    for(int64_t c=0; c<C; ++c){
                        for(int64_t kh=0; kh<Kh; ++kh){
                            for(int64_t kw=0; kw<Kw; ++kw){
                                // Map to input coordinates
                                int64_t ih = (h * Sh) + kh - Ph;
                                int64_t iw = (w * Sw) + kw - Pw;
                                // Boundary check for padding
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int64_t in_idx = (n*C*H*W) + (c*H*W) + (ih*W) + iw;
                                    int64_t w_idx = (oc*C*Kh*Kw) + (c*Kh*Kw) + (kh*Kw) + kw;
                                    float in_val = (in_idx >= 0 && in_idx < static_cast<int64_t>(input.size())) ? in_ptr[in_idx] : 0.0f;
                                    float w_val = (w_idx >= 0 && w_idx < static_cast<int64_t>(weights.size())) ? weight_ptr[w_idx] : 0.0f;
                                    sum += (in_val * w_val);
                                }
                            }
                        }
                    }
                    if(!bias.data.empty() && oc < static_cast<int64_t>(bias.size())){
                        sum += bias_ptr[oc];
                    }
                    int64_t out_idx = (n * C_out * H_out * W_out) + (oc * H_out * W_out) + (h * W_out) + w;
                    if (out_idx >= 0 && out_idx < static_cast<int64_t>(output.size())) {
                        out_ptr[out_idx] = sum;
                    }
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";

    return output;
}
