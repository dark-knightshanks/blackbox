#ifndef OPS_H
#define OPS_H
#include <vector>
#include <cmath>
#include <tensor.h>
#include <cstdint>

Tensor run_relu(const Tensor& input);
Tensor run_reshape(const Tensor& input, const Tensor& shape_tensor);
Tensor run_gemm(const Tensor& input, const Tensor& weights, const Tensor& bias);
Tensor run_mmaxpool2D(const Tensor& input,const std::vector<int64_t>& kernel, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads);
Tensor run_conv2D(const Tensor& input, const Tensor& weights, 
                    const Tensor& bias, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads);

#endif







