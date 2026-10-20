#include <iostream>
#include <cmath>
#include <algorithm>
#include "ops.h"
#include <chrono>
#include <numeric>

Tensor run_layernorm(Tensor& input, const Tensor& weights, const Tensor& bias) {

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    int batch, length, dims;
    batch = input.shape[0];
    length = input.shape[1];  // patches for ViT, tokens for LLM's
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
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < length; ++j) {
            int64_t offset = (i * length + j) * dims;
            float sum = 0.0f, sum_var = 0.0f, diff = 0.0f, var = 0.0f, mean = 0.0f;
            for (int k = 0; k < dims; ++k) {
                sum += in_ptr[offset + k];
            }
            mean = sum / dims;
            for (int k = 0; k < dims; ++k) {
                diff = in_ptr[offset + k] - mean;
                sum_var += diff * diff;
            }
            var = sum_var / dims;
            for (int k = 0; k < dims; ++k) {
                float norm = (in_ptr[offset + k] - mean) / (sqrt(var + 1e-5f));
                out_ptr[offset + k] = norm * weight_ptr[k] + (bias_ptr ? bias_ptr[k] : 0.0f);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Duration: " << duration << " us\n";

    return output;
}

Tensor run_rmsnorm(const Tensor& input, const Tensor& weights) {

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    int batch, length, dims;
    batch = input.shape[0];
    length = input.shape[1];  // patches for ViT, tokens for LLM's
    dims = input.shape[2];
    const float* in_ptr = reinterpret_cast<const float*>(input.raw_data());
    const float* weight_ptr = reinterpret_cast<const float*>(weights.raw_data());
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < length; ++j) {
            float sum, mean, rms;
            int64_t offset = (i * length + j) * dims;

            sum = 0.0f;
            for (int k = 0; k < dims; ++k) {
                sum += (in_ptr[offset + k] * in_ptr[offset + k]);
            }
            mean = sum / dims;
            rms = sqrt(mean + 1e-5f);
            for (int k = 0; k < dims; ++k) {
                out_ptr[offset + k] = weight_ptr[k] * ((in_ptr[offset + k]) / (rms));
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Duration: " << duration << " us\n";

    return output;
}

Tensor run_batchNorm(const Tensor& input, const Tensor& scale, const Tensor& B, const Tensor& mean,
                     const Tensor& var, float epsilon) {
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output;
    output.shape = input.shape;
    output.data.resize(input.byte_size());
    int num, channels, height, width;
    num = input.shape[0];
    channels = input.shape[1];
    height = input.shape[2];
    width = input.shape[3];
    int64_t dim = height * width;
    const float* in_ptr = reinterpret_cast<const float*>(input.raw_data());
    const float* gamma = reinterpret_cast<const float*>(scale.raw_data());
    const float* beta = reinterpret_cast<const float*>(B.raw_data());
    const float* pmu = reinterpret_cast<const float*>(mean.raw_data());
    const float* sigma_sq = reinterpret_cast<const float*>(var.raw_data());

    float* out_ptr = reinterpret_cast<float*>(output.data.data());
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            float inv_std = 1.0f / std::sqrt(sigma_sq[c] + epsilon);
            float scale_factor = gamma[c] * inv_std;
            float shift = beta[c] - (pmu[c] * scale_factor);
            int64_t offset = (n * channels + c) * dim;
            for (int64_t hw = 0; hw < dim; ++hw) {
                out_ptr[offset + hw] = (in_ptr[offset + hw] * scale_factor) + shift;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Duration: " << duration << " us\n";
    return output;
}

void LayerNormOperator::forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                                const std::vector<const uint8_t*>& weight_ptr,
                                const std::vector<size_t>& output_offset) {
    const float* in_ptr = reinterpret_cast<const float*>(arena + input_offset[0]);
    const float* w_ptr = reinterpret_cast<const float*>(weight_ptr[0]);
    const float* b_ptr = has_bias ? reinterpret_cast<const float*>(weight_ptr[1]) : nullptr;
    float* out_ptr = reinterpret_cast<float*>(arena + output_offset[0]);

    for (int64_t i = 0; i < batch; ++i) {
        for (int64_t j = 0; j < length; ++j) {
            int64_t offset = (i * length + j) * dims;
            float sum = 0.0f;
            for (int64_t k = 0; k < dims; ++k) {
                sum += in_ptr[offset + k];
            }
            float mean = sum / dims;
            float sum_var = 0.0f;
            for (int64_t k = 0; k < dims; ++k) {
                float diff = in_ptr[offset + k] - mean;
                sum_var += diff * diff;
            }
            float var = sum_var / dims;
            float inv_std = 1.0f / std::sqrt(var + epsilon);
            for (int64_t k = 0; k < dims; ++k) {
                float norm = (in_ptr[offset + k] - mean) * inv_std;
                out_ptr[offset + k] = norm * w_ptr[k] + (has_bias ? b_ptr[k] : 0.0f);
            }
        }
    }
}

void RMSNormOperator::forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                              const std::vector<const uint8_t*>& weight_ptr,
                              const std::vector<size_t>& output_offset) {
    const float* in_ptr = reinterpret_cast<const float*>(arena + input_offset[0]);
    const float* w_ptr = reinterpret_cast<const float*>(weight_ptr[0]);
    float* out_ptr = reinterpret_cast<float*>(arena + output_offset[0]);

    for (int64_t i = 0; i < batch; ++i) {
        for (int64_t j = 0; j < length; ++j) {
            int64_t offset = (i * length + j) * dims;
            float sum = 0.0f;
            for (int64_t k = 0; k < dims; ++k) {
                sum += in_ptr[offset + k] * in_ptr[offset + k];
            }
            float mean = sum / dims;
            float inv_rms = 1.0f / std::sqrt(mean + epsilon);
            for (int64_t k = 0; k < dims; ++k) {
                out_ptr[offset + k] = w_ptr[k] * (in_ptr[offset + k] * inv_rms);
            }
        }
    }
}

void BatchNormOperator::forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                                const std::vector<const uint8_t*>& weight_ptr,
                                const std::vector<size_t>& output_offset) {
    const float* in_ptr = reinterpret_cast<const float*>(arena + input_offset[0]);
    const float* gamma = reinterpret_cast<const float*>(weight_ptr[0]);
    const float* beta = reinterpret_cast<const float*>(weight_ptr[1]);
    const float* pmu = reinterpret_cast<const float*>(weight_ptr[2]);
    const float* sigma_sq = reinterpret_cast<const float*>(weight_ptr[3]);
    float* out_ptr = reinterpret_cast<float*>(arena + output_offset[0]);

    int64_t dim = height * width;

    for (int64_t n = 0; n < num; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            float inv_std = 1.0f / std::sqrt(sigma_sq[c] + epsilon);
            float scale_factor = gamma[c] * inv_std;
            float shift = beta[c] - (pmu[c] * scale_factor);
            int64_t offset = (n * channels + c) * dim;
            for (int64_t hw = 0; hw < dim; ++hw) {
                out_ptr[offset + hw] = (in_ptr[offset + hw] * scale_factor) + shift;
            }
        }
    }
}