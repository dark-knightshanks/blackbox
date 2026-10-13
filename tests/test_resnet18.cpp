#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <unordered_map>
#include "tensor.h"
#include "onnx.proto3.pb.h"
#include "engine.h"
#include "ops.h"
#include "test.h"

std::vector<test_cases> test = {
    {"tests/assets/resnet18/dog.bin", 258, "Samoyed"},
    {"tests/assets/resnet18/cat.bin", 285, "Egyptian Cat"}
};
/*to load the image from the assests folder*/
Tensor load_image(const std::string& filepath) {
    Tensor tensor;

    tensor.shape = {1, 3, 224, 224};
    constexpr size_t total_elements = 3*224*224;
    tensor.data.resize(total_elements*sizeof(float));
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

int main(){
    onnx::ModelProto model;
    std::ifstream input("resnet18.onnx", std::ios::binary);
    if(!input.is_open()){
        std::cerr<<"Could not load file!!"<<std::endl;
        return 1;
    }
    model.ParseFromIstream(&input);
    const onnx::GraphProto& graphproto = model.graph();
    loadinitializer(graphproto);

    int correct_count = 0;

    for (int i = 0; i < test.size() ; ++i) {
        
        try {
            Tensor input_image = load_image(test[i].filename);
            Tensor output_digit = run_inference(graphproto, input_image);
            int predicted = argmax(output_digit);
            float* out_ptr = reinterpret_cast<float*>(output_digit.data.data());

            bool is_correct = (predicted == test[i].expected_id);
            if (is_correct) {
                correct_count++;
            }

            std::cout << "File: " << test[i].filename 
                      << " | Expected: " << test[i].label_name 
                      << " | Predicted: " << predicted 
                      << " | Confidence Score: " << out_ptr[predicted]
                      << " | Status: " << (is_correct ? "PASS" : "FAIL") 
                      << "\n";
        } 
        catch (const std::exception& e) {
            std::cerr << "Error testing " << test[i].filename << ": " << e.what() << "\n";
        }
    }

    float accuracy = (static_cast<float>(correct_count) / test.size()) * 100.0f;

    std::cout << "\n================ Final Summary ================\n";
    std::cout << "Correct      : " << correct_count << "\n";
    std::cout << "Accuracy     : " << accuracy      << "\n";
    std::cout << "===============================================\n";
    return 0;
}
