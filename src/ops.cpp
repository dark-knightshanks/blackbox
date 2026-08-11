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