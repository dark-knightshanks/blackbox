# Mathematical Operations & OOP Operator Library

The operator library in `blackbox` is structured around an abstract, polymorphic `Operator` base class (`include/ops.h`), with zero-allocation kernels partitioned into modular domain files across `src/`:

```cpp
class Operator {
public:
    virtual ~Operator() = default;
    virtual void forward(uint8_t* arena,
                         const std::vector<size_t>& input_offset,
                         const std::vector<const uint8_t*>& weight_ptr,
                         const std::vector<size_t>& output_offset) = 0;
    virtual const std::string& name() const = 0;
};
```

---

## 1. Convolution & Spatial Pooling (`src/ops_conv.cpp`)

* **`Conv2DOperator`**:
  * Implements 4D spatial convolution (`[N, C, H, W]`).
  * Supports asymmetric padding, variable strides, and optional bias addition.
  * Direct pointer access into the Arena with zero-copy mmap filter weights.
* **`MaxPool2DOperator`**:
  * 2D spatial window max pooling. Pre-computes output height and width at instantiation.
* **`GlobalAvgPoolOperator`**:
  * Averages each spatial feature slice ($H \times W$) into $1 \times 1$ per channel.

---

## 2. Activation Functions (`src/ops_activations.cpp`)

* **`RELUOperator`**:
  * Rectified Linear Unit activation ($\max(0, x)$). Supports in-place execution (`in_offset == out_offset`).
* **`GELUOperator`**:
  * Gaussian Error Linear Unit with tanh approximation ($0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$), used in Vision Transformers and BERT.
* **`SILUOperator`**:
  * Sigmoid Linear Unit ($x \cdot \sigma(x)$), standard in modern vision backbones and LLMs.
* **`SoftmaxOperator`**:
  * Numerically stable 3-pass softmax ($\max$ subtraction, exponentiation + accumulation, normalization) along an arbitrary axis.

---

## 3. Normalization (`src/ops_norm.cpp`)

* **`BatchNormOperator`**:
  * Normalizes across spatial pixels ($H \times W$) per channel using 4 pre-trained mmap vectors: scale ($\gamma$), shift ($\beta$), mean ($\mu$), and variance ($\sigma^2$).
  * Supports in-place memory reuse.
* **`LayerNormOperator`**:
  * Normalizes across the hidden dimension (`dims`) per token/patch for Transformers (ViT).
* **`RMSNormOperator`**:
  * Root Mean Square Normalization without mean centering, used in modern LLMs (LLaMA).

---

## 4. Matrix Multiplication & Linear (`src/ops_matmul.cpp`)

* **`GemmOperator`**:
  * General Matrix Multiply for 2D classification heads and linear layers: $Y = X \cdot W^T + b$ (`transB = 1` supported).
* **`MatMulOperator`**:
  * 4D Batched Matrix Multiplication ($[B, \text{heads}, M, K] \times [B, \text{heads}, K, N]$) with broadcasting across batch and head dimensions, powering Multi-Head Attention.

---

## 5. Shape Operations (`src/ops_shape.cpp`)

* **`ADDOperator`**:
  * Elementwise addition and broadcasting addition, powering ResNet residual skip connections and Transformer positional embedding additions.
* **`ReshapeOperator` / `FlattenOperator`**:
  * Zero-cost shape reinterpretation. If `in_offset == out_offset`, operates as a true zero-overhead no-op.
* **`TransposeOperator`**:
  * Permutes 4D dimension order (e.g. $[0, 2, 1, 3]$ for attention head transformations or NCHW $\to$ NHWC).
* **`ConcatOperator`**:
  * Concatenates multiple tensors along an arbitrary axis using fast `std::memcpy` blocks.
* **`PadOperator`**:
  * Padds spatial borders with constant values.
