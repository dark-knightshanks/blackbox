# Mathematical Operations

The `ops.cpp` file contains all the mathematical and tensor manipulation functions required to execute the neural network layers.

```cpp
Tensor run_conv2D(const Tensor& input, const Tensor& weights, const Tensor& bias, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads);
```
***run_conv2D*** - performs 2D spatial convolution over an input tensor, handling asymmetric padding and strides, with safe optional bias addition.

```cpp
Tensor run_maxpool2D(const Tensor& input, const std::vector<int64_t>& kernel, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads);
```
***run_maxpool2D*** - applies a 2D max pooling filter over the input to downsample spatial dimensions.

```cpp
Tensor run_gemm(const Tensor& input, const Tensor& weights, const Tensor& bias, int transB);
```
***run_gemm*** - executes General Matrix Multiplication for linear/dense layers, supporting weight transposition (`transB`).

```cpp
Tensor run_relu(const Tensor& input);
```
***run_relu*** - applies the Rectified Linear Unit activation function, setting all negative values in the tensor to zero.

```cpp
Tensor run_reshape(const Tensor& input, const Tensor& shape);
```
***run_reshape*** - takes an input tensor and raw ONNX shape data, safely parsing it to reshape the output tensor (including inferring `-1` dimensions).

```cpp
int argmax(const Tensor& output);
```
***argmax*** - scans the final 1D output tensor and returns the index of the highest confidence score (used for final classification).
