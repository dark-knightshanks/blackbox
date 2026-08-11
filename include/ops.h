#ifndef OPS_H
#define OPS_H
#include <vector>
#include <cmath>
#include <tensor.h>
#include <cstdint>

Tensor run_relu(const Tensor& input);
Tensor run_reshape(const Tensor& input, const Tensor& shape_tensor);

#endif







