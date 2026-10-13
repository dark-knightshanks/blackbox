#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include "onnx.proto3.pb.h"

int main() {
    onnx::ModelProto model;
    std::ifstream input("vit-base-patch16-224.onnx", std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Could not open vit-base-patch16-224.onnx\n";
        return 1;
    }
    model.ParseFromIstream(&input);
    const onnx::GraphProto& graph = model.graph();

    std::set<std::string> op_types;
    for (const auto& node : graph.node()) {
        op_types.insert(node.op_type());
    }

    std::cout << "\n=== All Operator Types in ViT Model ===\n";
    for (const auto& op : op_types) {
        std::cout << "- " << op << "\n";
    }
    std::cout << "=======================================\n";
    return 0;
}
