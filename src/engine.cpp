#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "tensor.h"
#include "onnx.proto3.pb.h"
#include "engine.h"
#include "ops.h"

std::unordered_map<std::string, Tensor> weight;
/* loads tensor data along with their string name in a map */
static std::unordered_map<std::string, const uint8_t*> mmap_cache;

const uint8_t* get_mmap_buffer(const std::string& filepath) {
    if (mmap_cache.find(filepath) != mmap_cache.end()) {
        return mmap_cache[filepath];
    }
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Could not open external dats file for mmap");
    }
    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        close(fd);
        throw std::runtime_error("fstat failed:");
    }
    void* ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }
    const uint8_t* base = static_cast<const uint8_t*>(ptr);
    mmap_cache[filepath] = base;
    return base;
}

void loadinitializer(const onnx::GraphProto& graph) {
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const onnx::TensorProto& tensor_onnx = graph.initializer(i);
        std::string name = tensor_onnx.name();
        Tensor my_tensor;

        for (int j = 0; j < tensor_onnx.dims_size(); ++j) {
            my_tensor.shape.push_back(tensor_onnx.dims(j));
        }
        // 1. Raw byte buffer
        if (!tensor_onnx.raw_data().empty()) {
            my_tensor.data.resize(tensor_onnx.raw_data().size());
            float* ptr = reinterpret_cast<float*>(my_tensor.data.data());
            std::memcpy(ptr, tensor_onnx.raw_data().data(), tensor_onnx.raw_data().size());
        }
        // 2. External data
        else if (tensor_onnx.data_location() == onnx::TensorProto::EXTERNAL) {

            std::string location;
            size_t offset = 0;
            size_t length = 0;

            for (int j = 0; j < tensor_onnx.external_data_size(); ++j) {
                const auto& entry = tensor_onnx.external_data(j);

                if (entry.key() == "location") {
                    location = entry.value();
                } else if (entry.key() == "offset") {
                    offset = std::stoull(entry.value());
                } else if (entry.key() == "length") {
                    length = std::stoull(entry.value());
                }
            }
            const uint8_t* base = get_mmap_buffer(location);
            my_tensor.external_ptr = base + offset;
        }

        // 3. Repeated float field
        else if (tensor_onnx.float_data_size() > 0) {
            my_tensor.data.resize(tensor_onnx.float_data_size() * sizeof(float));
            float* ptr = reinterpret_cast<float*>(my_tensor.data.data());
            for (int j = 0; j < tensor_onnx.float_data_size(); ++j) {
                ptr[j] = tensor_onnx.float_data(j);
            }
        }

        // 4. Repeated int64 field (used for Reshape shapes)
        else if (tensor_onnx.int64_data_size() > 0) {
            my_tensor.data.resize(tensor_onnx.int64_data_size() * sizeof(int64_t));
            int64_t* ptr = reinterpret_cast<int64_t*>(my_tensor.data.data());

            std::memcpy(ptr, tensor_onnx.int64_data().data(),
                        tensor_onnx.int64_data_size() * sizeof(int64_t));
        }

        weight[name] = my_tensor;
        std::cout << "Loaded weight: " << name << " with shape [";
        for (auto d : my_tensor.shape)
            std::cout << d << " ";
        std::cout << "]\n";
    }
}

/*goes through each node in the Graphproto annd takes it op_type, input/output names and their
 * attribute data*/
std::vector<Node> graph(const onnx::GraphProto& graph) {
    std::vector<Node> parse;

    for (int i = 0; i < graph.node_size(); ++i) {
        const onnx::NodeProto& node = graph.node(i);
        Node current_node;
        current_node.op_type = node.op_type();
        for (int j = 0; j < node.input_size(); ++j) {
            current_node.input_names.push_back(node.input(j));
        }
        for (int j = 0; j < node.output_size(); ++j) {
            current_node.output_names.push_back(node.output(j));
        }
        for (int j = 0; j < node.attribute_size(); ++j) {
            const onnx::AttributeProto& attr = node.attribute(j);
            if (attr.ints_size() > 0) {
                for (int k = 0; k < attr.ints_size(); ++k) {
                    current_node.int_attributes[attr.name()].push_back(attr.ints(k));
                }
            }
        }
        parse.push_back(current_node);
    }
    return parse;
}

/*goes through each node and calls respective operation function*/
Tensor run_inference(const onnx::GraphProto& graph, const Tensor& input_image) {
    auto start = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, Tensor> inputs = weight;

    for (int i = 0; i < graph.input_size(); ++i) {
        std::string name = graph.input(i).name();
        if (weight.find(name) == weight.end()) {
            inputs[name] = input_image;
        }
    }
    if (graph.node_size() > 0 && graph.node(0).input_size() > 0) {
        inputs[graph.node(0).input(0)] = input_image;
    }
    auto get_tensor = [&](const std::string& name) -> const Tensor& {
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            throw std::runtime_error("Missing required tensor in map: '" + name + "'");
        }
        return it->second;
    };
    for (const auto& node : graph.node()) {
        const std::string& op_type = node.op_type();
        std::cout << "Executing Node: " << node.name() << " [" << op_type << "]\n";

        if (op_type == "Conv") {
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& weights = get_tensor(node.input(1));
            Tensor bias = (node.input_size() > 2 && inputs.find(node.input(2)) != inputs.end())
                              ? inputs[node.input(2)]
                              : Tensor();
            std::vector<int64_t> strides, pads, kernel_shape;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "strides") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        strides.push_back(attr.ints(i));
                    }
                } else if (attr.name() == "pads") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        pads.push_back(attr.ints(i));
                    }
                } else if (attr.name() == "kernel_shape") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        kernel_shape.push_back(attr.ints(i));
                    }
                }
            }

            inputs[node.output(0)] = run_conv2D(in, weights, bias, strides, pads);
        } else if (op_type == "Relu") {
            inputs[node.output(0)] = run_relu(inputs[node.input(0)]);
        } else if (op_type == "MaxPool") {
            const Tensor& in = get_tensor(node.input(0));
            std::vector<int64_t> strides, pads, kernel_shape;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "strides") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        strides.push_back(attr.ints(i));
                    }
                } else if (attr.name() == "pads") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        pads.push_back(attr.ints(i));
                    }
                } else if (attr.name() == "kernel_shape") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        kernel_shape.push_back(attr.ints(i));
                    }
                }
            }
            inputs[node.output(0)] = run_maxpool2D(in, kernel_shape, strides, pads);
        } else if (op_type == "Reshape") {
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& shape = get_tensor(node.input(1));
            inputs[node.output(0)] = run_reshape(in, shape);
        } else if (op_type == "Gemm") {
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& weights = get_tensor(node.input(1));
            Tensor bias = (node.input_size() > 2 && inputs.find(node.input(2)) != inputs.end())
                              ? inputs[node.input(2)]
                              : Tensor();
            // Extract transB attribute from ONNX node (defaults to 0 if not specified)
            int transB = 0;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "transB") {
                    transB = static_cast<int>(attr.i());
                }
            }
            inputs[node.output(0)] = run_gemm(in, weights, bias, transB);
        } else if (op_type == "BatchNormalization") {
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& scale = get_tensor(node.input(1));
            const Tensor& bias = get_tensor(node.input(2));
            const Tensor& mean = get_tensor(node.input(3));
            const Tensor& var = get_tensor(node.input(4));
            float epsilon = 1e-5f;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "epsilon") {
                    epsilon = attr.f();
                }
            }
            inputs[node.output(0)] = run_batchNorm(in, scale, bias, mean, var, epsilon);
        } else if (op_type == "GlobalAveragePool") {
            const Tensor& in = get_tensor(node.input(0));
            inputs[node.output(0)] = run_globalAvgPool(in);
        } else if (op_type == "Flatten") {
            const Tensor& in = get_tensor(node.input(0));
            int64_t axis = 1;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "axis") {
                    axis = attr.i();
                }
            }
            if (axis < 0)
                axis += in.shape.size();

            int64_t d0 = 1;
            for (int64_t i = 0; i < axis; ++i)
                d0 *= in.shape[i];
            int64_t d1 = 1;
            for (size_t i = axis; i < in.shape.size(); ++i)
                d1 *= in.shape[i];

            Tensor out = in;
            out.shape = {d0, d1};
            inputs[node.output(0)] = out;
        } else if (op_type == "Add") {
            const Tensor& input0 = get_tensor(node.input(0));
            const Tensor& input1 = get_tensor(node.input(1));
            inputs[node.output(0)] = run_add(input0, input1);
        } else if (op_type == "Concat") {
            std::vector<Tensor> concat_inputs;
            for (int i = 0; i < node.input_size(); ++i) {
                concat_inputs.push_back(get_tensor(node.input(i)));
            }
            int64_t axis = 1;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "axis") {
                    axis = attr.i();
                }
            }
            inputs[node.output(0)] = run_concat(concat_inputs, axis);
        } else if (op_type == "Pad") {
            const Tensor& in = get_tensor(node.input(0));
            std::vector<int64_t> pads;
            float constant_value = 0.0f;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "pads") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        pads.push_back(attr.ints(i));
                    }
                } else if (attr.name() == "value") {
                    constant_value = attr.f();
                }
            }
            if (pads.empty() && node.input_size() > 1 &&
                inputs.find(node.input(1)) != inputs.end()) {
                const Tensor& pads_tensor = inputs[node.input(1)];
                if (pads_tensor.data.size() == pads_tensor.size() * sizeof(int64_t)) {
                    const int64_t* ptr = reinterpret_cast<const int64_t*>(pads_tensor.raw_data());
                    for (size_t i = 0; i < pads_tensor.size(); ++i) {
                        pads.push_back(ptr[i]);
                    }
                }
            }
            inputs[node.output(0)] = run_pad(in, pads, constant_value);
        } else if (op_type == "MatMul") {
            const Tensor& A = get_tensor(node.input(0));
            const Tensor& B = get_tensor(node.input(1));
            inputs[node.output(0)] = run_matmul(A, B);
        } else if (op_type == "Softmax") {
            const Tensor& in = get_tensor(node.input(0));
            int64_t axis = -1;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "axis") {
                    axis = attr.i();
                }
            }
            inputs[node.output(0)] = run_softmax(in, axis);
        } else if (op_type == "Transpose") {
            const Tensor& in = get_tensor(node.input(0));
            std::vector<int64_t> perm;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "perm") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        perm.push_back(attr.ints(i));
                    }
                }
            }
            inputs[node.output(0)] = run_transpose(in, perm);
        } else if (op_type == "LayerNormalization") {
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& scale = get_tensor(node.input(1));
            Tensor bias = (node.input_size() > 2 && inputs.find(node.input(2)) != inputs.end())
                              ? inputs[node.input(2)]
                              : Tensor();
            inputs[node.output(0)] = run_layernorm(const_cast<Tensor&>(in), scale, bias);
        } else if (op_type == "Gelu") {
            const Tensor& in = get_tensor(node.input(0));
            inputs[node.output(0)] = run_GELU(in);
        } else if (op_type == "Silu") {
            const Tensor& in = get_tensor(node.input(0));
            inputs[node.output(0)] = run_SilU(in);
        }
    }
    std::string final_output_name = graph.output(0).name();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Duration: " << duration << " us\n";

    return inputs[final_output_name];
}

ExecutionGraph build_execution_graph(const onnx::GraphProto& graph,
                                     const std::vector<int64_t>& input_shape, size_t arena_size) {
    std::unordered_map<std::string, size_t> tensor_offsets;
    std::unordered_map<std::string, std::vector<int64_t>> tensor_shapes;

    std::string input_name = graph.input_size() > 0 ? graph.input(0).name() : "";
    if (graph.node_size() > 0 && graph.node(0).input_size() > 0) {
        if (input_name.empty() || weight.find(input_name) != weight.end()) {
            input_name = graph.node(0).input(0);
        }
    }

    tensor_offsets[input_name] = 0;
    tensor_shapes[input_name] = input_shape;

    int64_t input_elements = 1;
    for (int64_t d : input_shape) {
        input_elements *= d;
    }
    size_t current_offset = (input_elements * sizeof(float) + 63) & ~size_t(63);

    std::string final_output_name = graph.output_size() > 0 ? graph.output(0).name() : "";

    // Temporary graph object to add steps into
    std::vector<std::unique_ptr<Operator>> ops;
    std::vector<ExecutionStep> plan;

    for (const auto& node : graph.node()) {
        const std::string& op_type = node.op_type();

        if (op_type == "Conv") {
            const std::string& in_name = node.input(0);
            const std::string& w_name = node.input(1);
            const auto& in_shape = tensor_shapes[in_name];
            const auto& w_tensor = weight[w_name];

            int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
            int64_t C_out = w_tensor.shape[0];
            int64_t Kh = w_tensor.shape[2], Kw = w_tensor.shape[3];

            int64_t Sh = 1, Sw = 1, Ph = 0, Pw = 0;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "strides") {
                    if (attr.ints_size() > 0)
                        Sh = attr.ints(0);
                    if (attr.ints_size() > 1)
                        Sw = attr.ints(1);
                } else if (attr.name() == "pads") {
                    if (attr.ints_size() > 0)
                        Ph = attr.ints(0);
                    if (attr.ints_size() > 1)
                        Pw = attr.ints(1);
                }
            }

            int64_t H_out = (H + 2 * Ph - Kh) / Sh + 1;
            int64_t W_out = (W + 2 * Pw - Kw) / Sw + 1;
            std::vector<int64_t> out_shape = {N, C_out, H_out, W_out};

            size_t out_elements = N * C_out * H_out * W_out;
            size_t out_offset = (current_offset + 63) & ~size_t(63);
            current_offset = out_offset + out_elements * sizeof(float);

            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = out_shape;

            bool has_bias = node.input_size() > 2 && weight.find(node.input(2)) != weight.end();
            std::vector<const uint8_t*> weight_ptrs = {w_tensor.raw_data()};
            if (has_bias) {
                weight_ptrs.push_back(weight[node.input(2)].raw_data());
            }

            auto op = std::make_unique<Conv2DOperator>(N, C, H, W, C_out, Kh, Kw, Sh, Sw, Ph, Pw,
                                                       has_bias);
            ops.push_back(std::move(op));
            plan.push_back(
                {ops.back().get(), {tensor_offsets[in_name]}, weight_ptrs, {out_offset}});

        } else if (op_type == "BatchNormalization") {
            const std::string& in_name = node.input(0);
            const auto& in_shape = tensor_shapes[in_name];
            int64_t num = in_shape[0], channels = in_shape[1], height = in_shape[2],
                    width = in_shape[3];

            float epsilon = 1e-5f;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "epsilon")
                    epsilon = attr.f();
            }

            size_t out_elements = num * channels * height * width;
            // In-place reuse: can use same offset as input!
            size_t out_offset = tensor_offsets[in_name];
            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = in_shape;

            std::vector<const uint8_t*> weight_ptrs = {
                weight[node.input(1)].raw_data(), weight[node.input(2)].raw_data(),
                weight[node.input(3)].raw_data(), weight[node.input(4)].raw_data()};

            auto op = std::make_unique<BatchNormOperator>(num, channels, height, width, epsilon);
            ops.push_back(std::move(op));
            plan.push_back(
                {ops.back().get(), {tensor_offsets[in_name]}, weight_ptrs, {out_offset}});

        } else if (op_type == "Relu") {
            const std::string& in_name = node.input(0);
            const auto& in_shape = tensor_shapes[in_name];
            int64_t count = 1;
            for (int64_t d : in_shape)
                count *= d;

            // In-place reuse!
            size_t out_offset = tensor_offsets[in_name];
            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = in_shape;

            auto op = std::make_unique<RELUOperator>(count);
            ops.push_back(std::move(op));
            plan.push_back({ops.back().get(), {tensor_offsets[in_name]}, {}, {out_offset}});

        } else if (op_type == "MaxPool") {
            const std::string& in_name = node.input(0);
            const auto& in_shape = tensor_shapes[in_name];
            int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];

            int64_t Kh = 1, Kw = 1, Sh = 1, Sw = 1, Ph = 0, Pw = 0;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "kernel_shape") {
                    if (attr.ints_size() > 0)
                        Kh = attr.ints(0);
                    if (attr.ints_size() > 1)
                        Kw = attr.ints(1);
                } else if (attr.name() == "strides") {
                    if (attr.ints_size() > 0)
                        Sh = attr.ints(0);
                    if (attr.ints_size() > 1)
                        Sw = attr.ints(1);
                } else if (attr.name() == "pads") {
                    if (attr.ints_size() > 0)
                        Ph = attr.ints(0);
                    if (attr.ints_size() > 1)
                        Pw = attr.ints(1);
                }
            }

            int64_t H_out = (H + 2 * Ph - Kh) / Sh + 1;
            int64_t W_out = (W + 2 * Pw - Kw) / Sw + 1;
            std::vector<int64_t> out_shape = {N, C, H_out, W_out};

            size_t out_elements = N * C * H_out * W_out;
            size_t out_offset = (current_offset + 63) & ~size_t(63);
            current_offset = out_offset + out_elements * sizeof(float);

            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = out_shape;

            auto op = std::make_unique<MaxPool2DOperator>(N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw);
            ops.push_back(std::move(op));
            plan.push_back({ops.back().get(), {tensor_offsets[in_name]}, {}, {out_offset}});

        } else if (op_type == "GlobalAveragePool") {
            const std::string& in_name = node.input(0);
            const auto& in_shape = tensor_shapes[in_name];
            int64_t num = in_shape[0], channels = in_shape[1], height = in_shape[2],
                    width = in_shape[3];

            std::vector<int64_t> out_shape = {num, channels, 1, 1};
            size_t out_elements = num * channels;
            size_t out_offset = (current_offset + 63) & ~size_t(63);
            current_offset = out_offset + out_elements * sizeof(float);

            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = out_shape;

            auto op = std::make_unique<GlobalAvgPoolOperator>(num, channels, height, width);
            ops.push_back(std::move(op));
            plan.push_back({ops.back().get(), {tensor_offsets[in_name]}, {}, {out_offset}});

        } else if (op_type == "Flatten") {
            const std::string& in_name = node.input(0);
            const auto& in_shape = tensor_shapes[in_name];
            int64_t total = 1;
            for (int64_t d : in_shape)
                total *= d;

            std::vector<int64_t> out_shape = {in_shape[0], total / in_shape[0]};
            size_t out_offset = tensor_offsets[in_name];  // zero-cost alias!
            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = out_shape;

            auto op = std::make_unique<FlattenOperator>(total);
            ops.push_back(std::move(op));
            plan.push_back({ops.back().get(), {tensor_offsets[in_name]}, {}, {out_offset}});

        } else if (op_type == "Gemm") {
            const std::string& in_name = node.input(0);
            const std::string& w_name = node.input(1);
            const auto& in_shape = tensor_shapes[in_name];
            const auto& w_tensor = weight[w_name];

            int transB = 0;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "transB")
                    transB = static_cast<int>(attr.i());
            }

            int K = transB ? w_tensor.shape[1] : w_tensor.shape[0];
            int N = transB ? w_tensor.shape[0] : w_tensor.shape[1];
            int64_t in_total = 1;
            for (int64_t d : in_shape)
                in_total *= d;
            int M = (K > 0) ? static_cast<int>(in_total / K) : 1;

            std::vector<int64_t> out_shape = {M, N};
            size_t out_elements = M * N;
            size_t out_offset = (current_offset + 63) & ~size_t(63);
            current_offset = out_offset + out_elements * sizeof(float);

            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = out_shape;

            bool has_bias = node.input_size() > 2 && weight.find(node.input(2)) != weight.end();
            std::vector<const uint8_t*> weight_ptrs = {w_tensor.raw_data()};
            if (has_bias) {
                weight_ptrs.push_back(weight[node.input(2)].raw_data());
            }

            auto op = std::make_unique<GemmOperator>(M, K, N, transB, has_bias);
            ops.push_back(std::move(op));
            plan.push_back(
                {ops.back().get(), {tensor_offsets[in_name]}, weight_ptrs, {out_offset}});

        } else if (op_type == "Add") {
            const std::string& in0 = node.input(0);
            const std::string& in1 = node.input(1);
            const auto& s0 = tensor_shapes[in0];
            const auto& s1 = tensor_shapes[in1];

            int64_t count0 = 1;
            for (int64_t d : s0)
                count0 *= d;
            int64_t count1 = 1;
            for (int64_t d : s1)
                count1 *= d;

            int64_t big_count = std::max(count0, count1);
            int64_t small_count = std::min(count0, count1);

            size_t out_offset = (current_offset + 63) & ~size_t(63);
            current_offset = out_offset + big_count * sizeof(float);

            tensor_offsets[node.output(0)] = out_offset;
            tensor_shapes[node.output(0)] = (count0 >= count1) ? s0 : s1;

            auto op = std::make_unique<ADDOperator>(big_count, small_count);
            ops.push_back(std::move(op));
            plan.push_back(
                {ops.back().get(), {tensor_offsets[in0], tensor_offsets[in1]}, {}, {out_offset}});
        }
    }

    size_t out_offset = tensor_offsets[final_output_name];
    int64_t final_elements = 1;
    for (int64_t d : tensor_shapes[final_output_name]) {
        final_elements *= d;
    }
    size_t output_bytes = final_elements * sizeof(float);

    ExecutionGraph engine(arena_size, 0, out_offset, output_bytes);
    for (size_t i = 0; i < ops.size(); ++i) {
        engine.add_step(std::move(ops[i]), plan[i].in_offset, plan[i].weight_ptr,
                        plan[i].out_offset);
    }

    return engine;
}
