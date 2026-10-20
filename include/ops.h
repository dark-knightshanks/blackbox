#ifndef OPS_H
#define OPS_H
#include <vector>
#include <cmath>
#include <tensor.h>
#include <cstdint>
#include <cstddef>
#include <string>

class Operator {
public:
    virtual ~Operator() = default;
    virtual void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                         const std::vector<const uint8_t*>& weight_ptr,
                         const std::vector<size_t>& output_offset) = 0;

    virtual const std::string& name() const = 0;
};

class RELUOperator : public Operator {
private:
    int64_t count;  // no. of input float elements to process
    std::string op_name = "Relu";

public:
    RELUOperator(int64_t count) : count(count){};
    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;
    const std::string& name() const override { return op_name; };
};

class GELUOperator : public Operator {
private:
    int64_t count;
    std::string op_name = "Gelu";

public:
    GELUOperator(int64_t count) : count(count){};
    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;
    const std::string& name() const override { return op_name; };
};

class SILUOperator : public Operator {
private:
    int64_t count;
    std::string op_name = "Silu";

public:
    SILUOperator(int64_t count) : count(count){};
    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;
    const std::string& name() const override { return op_name; };
};

class SoftmaxOperator : public Operator {
private:
    std::vector<int64_t> shape;
    int64_t axis;
    std::string op_name = "Softmax";

public:
    SoftmaxOperator(const std::vector<int64_t>& shape, int64_t axis = -1)
        : shape(shape), axis(axis) {}
    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;
    const std::string& name() const override { return op_name; }
};

class ADDOperator : public Operator {
private:
    int64_t big_count;
    int64_t small_count;
    std::string op_name = "ADD";

public:
    ADDOperator(int64_t big_count, int64_t small_count)
        : big_count(big_count), small_count(small_count) {}
    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;
    const std::string& name() const override { return op_name; }
};

class ReshapeOperator : public Operator {
private:
    size_t count;
    std::string op_name = "Reshape";

public:
    explicit ReshapeOperator(size_t count) : count(count) {}

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class FlattenOperator : public Operator {
private:
    size_t count;
    std::string op_name = "Flatten";

public:
    explicit FlattenOperator(size_t count) : count(count) {}

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class TransposeOperator : public Operator {
private:
    std::vector<int64_t> input_shape;
    std::vector<int64_t> perm;
    std::string op_name = "Transpose";

public:
    explicit TransposeOperator(std::vector<int64_t> input_shape, std::vector<int64_t> perm)
        : input_shape(input_shape), perm(perm) {}

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class ConcatOperator : public Operator {
private:
    std::vector<std::vector<int64_t>> input_shapes;
    int64_t axis;
    std::string op_name = "Concat";

public:
    ConcatOperator(const std::vector<std::vector<int64_t>>& input_shapes, int64_t axis)
        : input_shapes(input_shapes), axis(axis) {}

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class PadOperator : public Operator {
private:
    std::vector<int64_t> in_shape;
    std::vector<int64_t> pads;
    float constant_val;
    std::string op_name = "Pad";

public:
    PadOperator(const std::vector<int64_t>& in_shape, const std::vector<int64_t>& pads,
                float constant_val = 0.0f)
        : in_shape(in_shape), pads(pads), constant_val(constant_val) {}

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class Conv2DOperator : public Operator {
private:
    int64_t N, C, H, W;
    int64_t C_out, Kh, Kw;
    int64_t Sh, Sw, Ph, Pw;
    int64_t H_out, W_out;
    bool has_bias;
    std::string op_name = "Conv2D";

public:
    Conv2DOperator(int64_t N, int64_t C, int64_t H, int64_t W, int64_t C_out, int64_t Kh,
                   int64_t Kw, int64_t Sh, int64_t Sw, int64_t Ph, int64_t Pw,
                   bool has_bias = false)
        : N(N), C(C), H(H), W(W), C_out(C_out), Kh(Kh), Kw(Kw), Sh(Sh), Sw(Sw), Ph(Ph), Pw(Pw),
          has_bias(has_bias) {
        H_out = (H + 2 * Ph - Kh) / Sh + 1;
        W_out = (W + 2 * Pw - Kw) / Sw + 1;
    }
    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class MaxPool2DOperator : public Operator {
private:
    int64_t N, C, H, W;
    int64_t Kh, Kw;
    int64_t Sh, Sw, Ph, Pw;
    int64_t H_out, W_out;
    std::string op_name = "MaxPool2D";

public:
    MaxPool2DOperator(int64_t N, int64_t C, int64_t H, int64_t W, int64_t Kh, int64_t Kw,
                      int64_t Sh, int64_t Sw, int64_t Ph, int64_t Pw)
        : N(N), C(C), H(H), W(W), Kh(Kh), Kw(Kw), Sh(Sh), Sw(Sw), Ph(Ph), Pw(Pw) {
        H_out = (H + 2 * Ph - Kh) / Sh + 1;
        W_out = (W + 2 * Pw - Kw) / Sw + 1;
    }
    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class GlobalAvgPoolOperator : public Operator {
private:
    int64_t num, channels, height, width;
    std::string op_name = "GlobalAveragePool";

public:
    GlobalAvgPoolOperator(int64_t num, int64_t channels, int64_t height, int64_t width)
        : num(num), channels(channels), height(height), width(width) {}

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class GemmOperator : public Operator {
private:
    int M, K, N;
    int transB;
    bool has_bias;
    std::string op_name = "Gemm";

public:
    GemmOperator(int M, int K, int N, int transB = 0, bool has_bias = false)
        : M(M), K(K), N(N), transB(transB), has_bias(has_bias) {}

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

class MatMulOperator : public Operator {
private:
    int64_t batch, heads, M, K, N;
    bool b_a_broadcast, b_b_broadcast;
    bool h_a_broadcast, h_b_broadcast;
    std::string op_name = "MatMul";

public:
    MatMulOperator(int64_t batch, int64_t heads, int64_t M, int64_t K, int64_t N,
                   bool b_a_broadcast = false, bool b_b_broadcast = false,
                   bool h_a_broadcast = false, bool h_b_broadcast = false)
        : batch(batch), heads(heads), M(M), K(K), N(N), b_a_broadcast(b_a_broadcast),
          b_b_broadcast(b_b_broadcast), h_a_broadcast(h_a_broadcast), h_b_broadcast(h_b_broadcast) {
    }

    void forward(uint8_t* arena, const std::vector<size_t>& input_offset,
                 const std::vector<const uint8_t*>& weight_ptr,
                 const std::vector<size_t>& output_offset) override;

    const std::string& name() const override { return op_name; }
};

constexpr double PI = 3.14159265358979323846;
Tensor run_relu(const Tensor& input);
Tensor run_GELU(const Tensor& input);
Tensor run_SilU(const Tensor& input);
int argmax(const Tensor& ouput);
Tensor run_reshape(const Tensor& input, const Tensor& shape_tensor);
Tensor run_gemm(const Tensor& input, const Tensor& weights, const Tensor& bias, int transB = 0);
Tensor run_maxpool2D(const Tensor& input, const std::vector<int64_t>& kernel,
                     const std::vector<int64_t>& strides, const std::vector<int64_t>& pads);
Tensor run_conv2D(const Tensor& input, const Tensor& weights, const Tensor& bias,
                  const std::vector<int64_t>& strides, const std::vector<int64_t>& pads);

Tensor run_layernorm(Tensor& input, const Tensor& weights, const Tensor& bias);
Tensor run_rmsnorm(const Tensor& input, const Tensor& weights);
Tensor run_transpose(const Tensor& input, const std::vector<int64_t>& perm);
Tensor run_softmax(const Tensor& input, int64_t axis = -1);
Tensor run_matmul(const Tensor& A, const Tensor& B);
Tensor run_batchNorm(const Tensor& input, const Tensor& scale, const Tensor& B, const Tensor& mean,
                     const Tensor& var, float epsilon);
Tensor run_globalAvgPool(const Tensor& input);
Tensor run_add(const Tensor& a, const Tensor& b);
Tensor run_pad(const Tensor& input, const std::vector<int64_t>& pads, float constant_val = 0.00f);
Tensor run_concat(const std::vector<Tensor>& inputs, int64_t axis);

#endif
