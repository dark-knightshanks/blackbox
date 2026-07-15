#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include "include/onnx.proto3.pb.h"
#include "include/Tensor.h"

std::unordered_map<std::string, Tensor> weight;// to store name, data and shape of weights, biases and inputs/outputs

/* loads tensor data alsong with their string name in a hap map*/
void loadinitializer(const onnx::GraphProto& graph){
    for (int i=0; i<graph.initializer_size(); ++i){
        const onnx::TensorProto& tensor_onnx = graph.initializer(i);
        std::string name = tensor_onnx.name();
        Tensor my_tensor;
        for(int j = 0; j<tensor_onnx.dims_size(); ++j){
            my_tensor.shape.push_back(tensor_onnx.dims(j));
        }
        // Allocate space in your flat C++ vector
        my_tensor.data.resize(my_tensor.size());

        // ONNX raw_data stores floats as strings of binary bytes
        if (!tensor_onnx.raw_data().empty()) {
            std::memcpy(my_tensor.data.data(), 
                        tensor_onnx.raw_data().data(), 
                        tensor_onnx.raw_data().size());
        }
        //Save it into the map by its string name identifier
        weight[name] = my_tensor;
        
        std::cout << "Loaded weight: " << name << " with shape [";
        for(auto d : my_tensor.shape) std::cout << d << " ";
        std::cout << "]\n";
    }
}