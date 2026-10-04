# Mathematical Operations

<<<<<<< HEAD
The operator library in `src/` contains mathematical and tensor manipulation functions partitioned into modular domains:

### 1. Convolution & Pooling (`src/ops_conv.cpp`)
=======
The `ops.cpp` file contains all the mathematical and tensor manipulation functions required to execute the neural network layers.
>>>>>>> origin/main

```cpp
Tensor run_conv2D(const Tensor& input, const Tensor& weights, const Tensor& bias, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads);
```
<<<<<<< HEAD
* ***run_conv2D*** - Performs 4D spatial convolution (`[N, C, H, W]`), handling asymmetric padding and strides, with zero-copy memory-mapped weights and optional bias addition.
=======
***run_conv2D*** - performs 2D spatial convolution over an input tensor, handling asymmetric padding and strides, with safe optional bias addition.
>>>>>>> origin/main

```cpp
Tensor run_maxpool2D(const Tensor& input, const std::vector<int64_t>& kernel, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads);
```
<<<<<<< HEAD
* ***run_maxpool2D*** - Applies a 2D max pooling filter over spatial dimensions.

---

### 2. Activations (`src/ops_activations.cpp`)

```cpp
Tensor run_relu(const Tensor& input);
Tensor run_GELU(const Tensor& input);
Tensor run_SilU(const Tensor& input);
Tensor run_softmax(const Tensor& input, int64_t axis);
```
* ***run_relu*** - Rectified Linear Unit activation (`max(0, x)`).
* ***run_GELU*** - Gaussian Error Linear Unit using the tanh approximation.
* ***run_SilU*** - Sigmoid Linear Unit ($x \cdot \sigma(x)$), used in modern LLMs and vision backbones.
* ***run_softmax*** - Numerically stable 3-pass softmax along a specified axis.

---

### 3. Normalization (`src/ops_norm.cpp`)

```cpp
Tensor run_layernorm(Tensor& input, const Tensor& weights, const Tensor& bias);
Tensor run_rmsnorm(const Tensor& input, const Tensor& weights);
```
* ***run_layernorm*** - Layer Normalization (mean subtraction + variance normalization + affine scale/bias).
* ***run_rmsnorm*** - Root Mean Square Normalization without mean tracking, used in LLaMA-style models.

---

### 4. Linear & Matrix Multiplication (`src/ops_matmul.cpp`)

```cpp
Tensor run_gemm(const Tensor& input, const Tensor& weights, const Tensor& bias, int transB);
Tensor run_matmul(const Tensor& A, const Tensor& B);
```
* ***run_gemm*** - General Matrix Multiplication for 2D linear/dense layers with `transB` support.
* ***run_matmul*** - 4D Batched Matrix Multiplication ($[B, H, M, K] \times [B, H, K, N] \to [B, H, M, N]$) with batch and head broadcasting.

---

### 5. Shape Operations (`src/ops_shape.cpp`)

```cpp
Tensor run_reshape(const Tensor& input, const Tensor& shape);
Tensor run_transpose(const Tensor& input, const std::vector<int64_t>& perm);
int argmax(const Tensor& output);
```
* ***run_reshape*** - Zero-copy pointer view reshape supporting inferred `-1` dimensions and 64-bit ONNX target shape tensors.
* ***run_transpose*** - 4D dimension permutation.
* ***argmax*** - Returns the index of the highest score for classification.
=======
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
>>>>>>> origin/main
