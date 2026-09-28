#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>
#include <numeric>


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

Tensor run_globalAvgPool(const Tensor& input){
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    int64_t num, channels, height, width;
    num = input.shape[0];
    channels = input.shape[1];
    height = input.shape[2];
    width = input.shape[3];

    output.shape = {num, channels, 1, 1};
    output.data.resize(output.byte_size());

    const float* in_ptr = reinterpret_cast<const float*>(input.data.data());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    float spatialSize = static_cast<float>(height*width);

    for(int64_t n = 0; n < num; ++n){
        for(int64_t c = 0 ; c < channels ; ++c){
            float sum = 00.0f;
            int64_t offset = (n*channels + c)*height*width;
            for(int64_t hw = 0 ; hw <  height*width ; ++hw){
                sum += in_ptr[offset + hw];
            }
            out_ptr[n*channels + c] = sum/spatialSize;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout<<"Duration: "<< duration << " us\n";
    return output;
}