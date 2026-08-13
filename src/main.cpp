#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include "onnx.proto3.pb.h"
#include "tensor.h"
#include "include.h"
#include "ops.h"

// to store name, data and shape of weights, biases and inputs/outputs
std::unordered_map<std::string, Tensor> weight;

/* loads tensor data along with their string name in a map */
void loadinitializer(const onnx::GraphProto& graph){
    for (int i=0; i<graph.initializer_size(); ++i){
        const onnx::TensorProto& tensor_onnx = graph.initializer(i);
        std::string name = tensor_onnx.name();
        Tensor my_tensor;

        for(int j = 0; j<tensor_onnx.dims_size(); ++j){
            my_tensor.shape.push_back(tensor_onnx.dims(j));
        }
        // 1. Raw byte buffer
        if (!tensor_onnx.raw_data().empty()) {
            my_tensor.data.resize(
                tensor_onnx.raw_data().size() / sizeof(float)
            );
            std::memcpy(
                my_tensor.data.data(),
                tensor_onnx.raw_data().data(),
                tensor_onnx.raw_data().size()
            );
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
                }
                else if (entry.key() == "offset") {
                    offset = std::stoull(entry.value());
                }
                else if (entry.key() == "length") {
                    length = std::stoull(entry.value());
                }
            }
            std::ifstream file(location, std::ios::binary);
            if (!file) {
                throw std::runtime_error(
                    "Could not open external data file: " + location
                );
            }
            file.seekg(offset);
            my_tensor.data.resize(length / sizeof(float));
            file.read(
                reinterpret_cast<char*>(my_tensor.data.data()),
                length
            );
            if (!file) {
                throw std::runtime_error(
                    "Failed to read external data for tensor: " + name
                );
            }
        }

        // 3. Repeated float field
        else if (tensor_onnx.float_data_size() > 0) {
            for (int j = 0; j < tensor_onnx.float_data_size(); ++j) {
                my_tensor.data.push_back(tensor_onnx.float_data(j));
            }
        }

        // 4. Repeated int64 field (used for Reshape shapes)
        else if (tensor_onnx.int64_data_size() > 0) {
            my_tensor.data.resize(
                tensor_onnx.int64_data_size() * sizeof(int64_t) / sizeof(float)
            );

            std::memcpy(
                my_tensor.data.data(),
                tensor_onnx.int64_data().data(),
                tensor_onnx.int64_data_size() * sizeof(int64_t)
            );
        }

        weight[name] = my_tensor;
        std::cout << "Loaded weight: " << name << " with shape [";
        for(auto d : my_tensor.shape) std::cout << d << " ";
        std::cout << "]\n";
    }
}

/*goes through each node in the Graphproto annd takes it op_type, input/output names and their attribute data*/
std::vector<Node> graph(const onnx::GraphProto& graph){
    std::vector<Node> parse;

    for (int i=0; i<graph.node_size(); ++i){
        const onnx::NodeProto& node = graph.node(i);
        Node current_node;
        current_node.op_type = node.op_type();
        for (int j=0; j<node.input_size(); ++j){
            current_node.input_names.push_back(node.input(j));
        }
        for (int j=0; j<node.output_size(); ++j){
            current_node.output_names.push_back(node.output(j));
        }
        for (int j=0; j<node.attribute_size(); ++j){
            const onnx::AttributeProto& attr = node.attribute(j);
            if(attr.name() == "strides" || attr.name() == "pads" || attr.name() == "kernel_shape"){
               for(int k=0; k<attr.ints_size(); ++k){
                current_node.int_attributes[attr.name()].push_back(attr.ints(k));
               } 
            }
        }  
        parse.push_back(current_node);
    }
    return parse;
}

/*to load the image from the assests folder*/
Tensor load_image(const std::string& filepath) {
    Tensor tensor;

    tensor.shape = {1, 1, 28, 28};
    constexpr size_t total_elements = 28 * 28;
    tensor.data.resize(total_elements);
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open image file: " + filepath
        );
    }
    file.read(
        reinterpret_cast<char*>(tensor.data.data()),
        total_elements * sizeof(float)
    );
    if (!file) {
        throw std::runtime_error(
            "Failed to read image: " + filepath
        );
    }
    file.close();
    return tensor;
}

/*goes through each node and calls respective operation function*/
Tensor run_inference(const onnx::GraphProto& graph, const Tensor& input_image){
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
    for(const auto& node : graph.node()){
        const std::string& op_type = node.op_type();
        std::cout << "Executing Node: " << node.name() << " [" << op_type << "]\n";

        if(op_type == "Conv"){
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& weights = get_tensor(node.input(1));
            Tensor bias = (node.input_size() > 2 && inputs.find(node.input(2)) != inputs.end()) 
                           ? inputs[node.input(2)] : Tensor();
            std::vector<int64_t> strides, pads, kernel_shape;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "strides") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        strides.push_back(attr.ints(i));
                    }
                } 
                else if (attr.name() == "pads") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        pads.push_back(attr.ints(i));
                    }
                } 
                else if (attr.name() == "kernel_shape") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        kernel_shape.push_back(attr.ints(i));
                    }
                }
            }

            inputs[node.output(0)] = run_conv2D(in, weights, bias, strides, pads);
        }
        else if (op_type == "Relu"){
            inputs[node.output(0)] = run_relu(inputs[node.input(0)]);
        }
        else if (op_type == "MaxPool"){
            const Tensor& in = get_tensor(node.input(0));
            std::vector<int64_t> strides, pads, kernel_shape;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "strides") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        strides.push_back(attr.ints(i));
                    }
                } 
                else if (attr.name() == "pads") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        pads.push_back(attr.ints(i));
                    }
                } 
                else if (attr.name() == "kernel_shape") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        kernel_shape.push_back(attr.ints(i));
                    }
                }
            }
            inputs[node.output(0)] = run_maxpool2D(in, kernel_shape, strides, pads);
        }
        else if(op_type == "Reshape"){
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& shape = get_tensor(node.input(1));
            inputs[node.output(0)] = run_reshape(in, shape);
        }
        else if(op_type == "Gemm"){
            const Tensor& in = get_tensor(node.input(0));
            const Tensor& weights = get_tensor(node.input(1));
            Tensor bias = (node.input_size() > 2 && inputs.find(node.input(2)) != inputs.end()) 
                           ? inputs[node.input(2)] : Tensor();
            // Extract transB attribute from ONNX node (defaults to 0 if not specified)
            int transB = 0;
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "transB") {
                    transB = static_cast<int>(attr.i());
                }
            }
            inputs[node.output(0)] = run_gemm(in, weights, bias, transB);
        }
    }
    std::string final_output_name = graph.output(0).name();
    return inputs[final_output_name];
}

int main(){
    onnx::ModelProto model;
    std::ifstream input("model.onnx", std::ios::binary);
    if(!input.is_open()){
        std::cerr<<"Could not load file!!"<<std::endl;
        return 1;
    }
    model.ParseFromIstream(&input);
    const onnx::GraphProto& graphproto = model.graph();
    loadinitializer(graphproto);

    int correct_count = 0;
    int total_images = 10;

    for (int expected = 0; expected < total_images; ++expected) {
        std::string filename = "assets/digit_" + std::to_string(expected) + ".bin";

        try {
            Tensor input_image = load_image(filename);
            Tensor output_digit = run_inference(graphproto, input_image);
            int predicted = argmax(output_digit);

            bool is_correct = (predicted == expected);
            if (is_correct) {
                correct_count++;
            }

            std::cout << "File: " << filename 
                      << " | Expected: " << expected 
                      << " | Predicted: " << predicted 
                      << " | Confidence Score: " << output_digit.data[predicted]
                      << " | Status: " << (is_correct ? "PASS" : "FAIL") 
                      << "\n";
        } 
        catch (const std::exception& e) {
            std::cerr << "Error testing " << filename << ": " << e.what() << "\n";
        }
    }

    float accuracy = (static_cast<float>(correct_count) / total_images) * 100.0f;

    std::cout << "\n================ Final Summary ================\n";
    std::cout << "Total Tested : " << total_images << "\n";
    std::cout << "Correct      : " << correct_count << "\n";
    std::cout << "Accuracy     : " << accuracy << "%\n";
    std::cout << "===============================================\n";
    return 0;
}