#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"


Tensor run_relu(const Tensor& input){
    Tensor output;
    output.shape = input.shape;
    output.data.resize(input.size());
    for(int i = 0; i<input.size();++i){
        output.data[i]=std::max(0.0f, input.data[i]);
    }
    return output;
}

Tensor run_reshape(const Tensor& input, const Tensor& shape) {
    Tensor output;
    output.data = input.data; // Directly copies the raw data

    int64_t total_elements = input.size();
    int64_t known_product = 1;
    int minus_one_index = -1;

    // Loop through target dimensions inside shape.data
    for (size_t i = 0; i < shape.data.size(); ++i) {
        int64_t target_dim = static_cast<int64_t>(shape.data[i]);

        if (target_dim == 0) {
            // Rule 0: Copy dimension from input shape
            int64_t dim = input.shape[i];
            output.shape.push_back(dim);
            known_product *= dim;
        } 
        else if (target_dim == -1) {
            // Rule -1: Remember index to solve later
            minus_one_index = i;
            output.shape.push_back(-1); // Temporary placeholder
        } 
        else {
            // Explicit dimension size
            output.shape.push_back(target_dim);
            known_product *= target_dim;
        }
    }

    // Solve for -1 if it was present
    if (minus_one_index != -1) {
        output.shape[minus_one_index] = total_elements / known_product;
    }

    return output;
}

Tensor run_gemm(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    Tensor output;
    int K = weights.shape[1];
    int N = weights.shape[0];
    int M = input.shape[0];

    output.shape = {M, N};
    output.data.assign(M * N, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int i1 = (i * K) + k;
                int i2 = (j * K) + k;
                sum += (input.data[i1] * weights.data[i2]); // Fix 1: input vs weights
            }

            // Fix 2: Add bias
            if (!bias.data.empty()) {
                sum += bias.data[j];
            }

            // Fix 3: Write sum into output tensor
            int out_index = (i * N) + j;
            output.data[out_index] = sum;
        }
    }

    return output;
}

Tensor run_mmaxpool2D(const Tensor& input,const std::vector<int64_t>& kernel, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads){
    Tensor output;
    int64_t N = input.shape[0];
    int64_t C = input.shape[1];
    int64_t H = input.shape[2];
    int64_t W = input.shape[3];
    int64_t Kh = kernel[0];
    int64_t Kw = kernel[1];
    int64_t Sh = strides.empty() ? 1 : strides[0];
    int64_t Sw = strides.empty() ? 1 : strides[1];
    int64_t Ph = pads.empty() ? 0 : pads[0];
    int64_t Pw = pads.empty() ? 0 : pads[1];

    int64_t H_out = (H + 2 * Ph - Kh) / Sh + 1;
    int64_t W_out = (W + 2 * Pw - Kw) / Sw + 1;
    output.shape = {N, C, H_out, W_out};
    output.data.resize(N * C * H_out * W_out);

    for(int64_t n=0; n<N; ++n){
        for(int64_t c=0; c<C; ++c){
            for(int64_t h=0; h<H_out; ++h){
                for(int64_t w=0; w<W_out; ++w){
                    float max_val = -INFINITY;
                    for(int64_t kh=0; kh<Kh; ++kh){
                        for(int64_t kw=0; kw<Kw; ++kw){
                            // Map to input coordinates
                            int64_t ih = (h * Sh) + kh - Ph;
                            int64_t iw = (w * Sw) + kw - Pw;

                            // Boundary check for padding
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t in_idx = (n * C * H * W) + (c * H * W) + (ih * W) + iw;
                                max_val = std::max(max_val, input.data[in_idx]);
                            }
                        }
                    }
                    // Save max result to output tensor
                    int64_t out_idx = (n * C * H_out * W_out) + (c * H_out * W_out) + (h * W_out) + w;
                    output.data[out_idx] = max_val;
                }
            }
        }
    }
    return output;
}

Tensor run_conv2D(const Tensor& input, const Tensor& weights, 
                    const Tensor& bias, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads){
    Tensor output;
    int64_t N = input.shape[0];
    int64_t C = input.shape[1];
    int64_t H = input.shape[2];
    int64_t W = input.shape[3];
    int64_t C_out = weights.shape[0];
    int64_t Kh = weights.shape[2];
    int64_t Kw = weights.shape[3];
    int64_t Sh = strides.empty() ? 1 : strides[0];
    int64_t Sw = strides.empty() ? 1 : strides[1];
    int64_t Ph = pads.empty() ? 0 : pads[0];
    int64_t Pw = pads.empty() ? 0 : pads[1];

    int64_t H_out = (H + 2 * Ph - Kh) / Sh + 1;
    int64_t W_out = (W + 2 * Pw - Kw) / Sw + 1;

    output.shape = {N, C_out, H_out, W_out};
    output.data.resize(N*C_out*H_out*W_out, 0.0f);

    for(int64_t n=0; n<N; ++n){
        for(int64_t oc=0; oc<C_out; ++oc){
            for(int64_t h=0; h<H_out; ++h){
                for(int64_t w=0; w<W_out; ++w){
                    float sum = 0.0f;
                    for(int c=0; c<C; ++c){
                        for(int64_t kh=0; kh<Kh; ++kh){
                            for(int64_t kw=0; kw<Kw; ++kw){
                                // Map to input coordinates
                                int64_t ih = (h * Sh) + kh - Ph;
                                int64_t iw = (w * Sw) + kw - Pw;
                                // Boundary check for padding
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int64_t in_idx = (n*C*H*W) + (c*H*W) + (ih*W) + iw;
                                    int64_t w_idx = (oc*C*Kh*Kw) + (c*Kh*Kw) + (kh*Kw) + kw;
                                    sum += (input.data[in_idx]*weights.data[w_idx]);
                                }
                            }
                        }
                    }
                    if(!bias.data.empty()){
                        sum += bias.data[oc];
                    }
                    int64_t out_idx = (n * C_out * H_out * W_out) + (oc * H_out * W_out) + (h * W_out) + w;
                    output.data[out_idx] = sum;
                }
            }
        }
    }
    return output;
}