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
