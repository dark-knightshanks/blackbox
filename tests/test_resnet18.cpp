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

struct test_cases {
    std::string filename;
    int expected_id;
    std::string label_name;
};

std::vector<test_cases> test = {{"tests/assets/resnet18/dog.bin", 258, "Samoyed"},
                                {"tests/assets/resnet18/cat.bin", 285, "Egyptian Cat"}};
/*to load the image from the assests folder*/
Tensor load_image(const std::string& filepath) {
    Tensor tensor;

    tensor.shape = {1, 3, 224, 224};
    constexpr size_t total_elements = 3 * 224 * 224;
    tensor.data.resize(total_elements * sizeof(float));
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + filepath);
    }
    file.read(reinterpret_cast<char*>(tensor.data.data()), total_elements * sizeof(float));
    if (!file) {
        throw std::runtime_error("Failed to read image: " + filepath);
    }
    file.close();
    return tensor;
}

int main() {
    onnx::ModelProto model;
    std::ifstream input("resnet18.onnx", std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Could not load file!!" << std::endl;
        return 1;
    }
    model.ParseFromIstream(&input);
    const onnx::GraphProto& graphproto = model.graph();
    loadinitializer(graphproto);

    std::cout << "\n--- Building ExecutionGraph (Compiling ONNX to Arena Steps) ---\n";
    ExecutionGraph engine = build_execution_graph(graphproto, {1, 3, 224, 224}, 50 * 1024 * 1024);
    std::cout << "--- Build Complete! Running zero-malloc Arena inference ---\n\n";

    int correct_count = 0;

    for (size_t i = 0; i < test.size(); ++i) {
        try {
            Tensor input_image = load_image(test[i].filename);
            std::vector<float> output_logits(1000);

            engine.run(reinterpret_cast<const float*>(input_image.raw_data()),
                       input_image.byte_size(), output_logits.data());

            int predicted = 0;
            float max_score = output_logits[0];
            for (int c = 1; c < 1000; ++c) {
                if (output_logits[c] > max_score) {
                    max_score = output_logits[c];
                    predicted = c;
                }
            }

            bool is_correct = (predicted == test[i].expected_id);
            if (is_correct) {
                correct_count++;
            }

            std::cout << "File: " << test[i].filename << " | Expected: " << test[i].label_name
                      << " | Predicted: " << predicted
                      << " | Confidence Score: " << output_logits[predicted]
                      << " | Status: " << (is_correct ? "PASS" : "FAIL") << "\n\n";
        } catch (const std::exception& e) {
            std::cerr << "Error testing " << test[i].filename << ": " << e.what() << "\n";
        }
    }

    float accuracy = (static_cast<float>(correct_count) / test.size()) * 100.0f;

    std::cout << "\n================ Final Summary ================\n";
    std::cout << "Correct      : " << correct_count << "\n";
    std::cout << "Accuracy     : " << accuracy << "\n";
    std::cout << "===============================================\n";
    return 0;
}
