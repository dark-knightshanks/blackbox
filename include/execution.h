#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cstring>
#include <chrono>
#include "ops.h"
#include "arena.h"

struct ExecutionStep {
    Operator* op;
    std::vector<size_t> in_offset;
    std::vector<const uint8_t*> weight_ptr;
    std::vector<size_t> out_offset;
};

class ExecutionGraph {
private:
    MemoryArena arena;
    std::vector<std::unique_ptr<Operator>> ops;
    std::vector<ExecutionStep> plan;
    size_t input_offset;
    size_t output_offset;
    size_t output_bytes;

public:
    explicit ExecutionGraph(size_t arena_size, size_t input_offsets, size_t output_offsets,
                            size_t output_bytes)
        : arena(arena_size), input_offset(input_offsets), output_offset(output_offsets),
          output_bytes(output_bytes){};

    ExecutionGraph(const ExecutionGraph&) = delete;
    ExecutionGraph& operator=(const ExecutionGraph&) = delete;

    void add_step(std::unique_ptr<Operator> op, std::vector<size_t> input_offsets,
                  std::vector<const uint8_t*> weight_ptrs, std::vector<size_t> output_offsets) {
        ops.push_back(std::move(op));  // transfers ownership to vector
        plan.push_back({ops.back().get(), input_offsets, weight_ptrs, output_offsets});
    }
    void run(const float* input, size_t in_bytes, float* output) {
        auto total_start = std::chrono::high_resolution_clock::now();

        memcpy(arena.base() + input_offset, input, in_bytes);

        for (const auto& step : plan) {
            auto step_start = std::chrono::high_resolution_clock::now();

            step.op->forward(arena.base(), step.in_offset, step.weight_ptr, step.out_offset);

            auto step_end = std::chrono::high_resolution_clock::now();
            auto step_us =
                std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start)
                    .count();
            std::cout << "Executing: " << step.op->name() << " [" << step_us << " us]\n";
        }

        memcpy(output, arena.base() + output_offset, output_bytes);

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
        std::cout << "Total Inference Time: " << total_us / 1000.0f << " ms\n";
    }
};
