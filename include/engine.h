#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct Node{
std::string op_type;
std::vector<std::string> input_names;
std::vector<std::string> output_names;
std::unordered_map<std::string, std::vector<int64_t>> int_attributes;
};

void loadinitializer(const onnx::GraphProto& graph);
std::vector<Node> graph(const onnx::GraphProto& graph);
Tensor run_inference(const onnx::GraphProto& graph, const Tensor& input_image);