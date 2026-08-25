#ifndef OPS_H
#define OPS_H
#include <vector>
#include <cmath>
#include <tensor.h>
#include <cstdint>

constexpr double PI = 3.14159265358979323846;
Tensor run_relu(const Tensor& input);
Tensor run_GELU(const Tensor& input);
Tensor run_SilU(const Tensor& input);
int argmax(const Tensor& ouput);
Tensor run_reshape(const Tensor& input, const Tensor& shape_tensor);
Tensor run_gemm(const Tensor& input, const Tensor& weights, const Tensor& bias, int transB = 0);
Tensor run_maxpool2D(const Tensor& input,const std::vector<int64_t>& kernel, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads);
Tensor run_conv2D(const Tensor& input, const Tensor& weights, 
                    const Tensor& bias, 
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& pads);

Tensor run_layernorm(Tensor& input, const Tensor& weights, const Tensor& bias);
Tensor run_rmsnorm(const Tensor& input, const Tensor& weights);
Tensor run_transpose(const Tensor& input, const std::vector<int64_t>& perm);
Tensor run_softmax(const Tensor& input, int64_t axis = -1);
Tensor run_matmul(const Tensor& A, const Tensor& B);

#endif







