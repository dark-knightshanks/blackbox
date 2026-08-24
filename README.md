# blackbox

A from-scratch ML inference engine written in C++17 — built to understand every layer of the stack, from model parsing to quantized compute.

Currently runs ONNX CNN models on CPU with FP32 precision. Being developed into a general-purpose inference library with quantization, layout optimization, and multi-backend support.

## Building

**Prerequisites**: `g++` (C++17), `libprotobuf-dev`, `protobuf-compiler`

First, if you haven't already, generate the C++ Protobuf bindings from the ONNX proto file:
```bash
protoc --cpp_out=. onnx.proto3
# This will generate onnx.proto3.pb.cc and onnx.proto3.pb.h
```

Then build the project:
```bash
make            # builds bin/onnx_engine
./bin/onnx_engine
make clean      # removes build artifacts
```

Expects `model.onnx` + `model.onnx.data` in project root and test images in `tests/assets/`.

## What's Done

- **Model loading**: ONNX format via protobuf (raw data, external data, float/int64 fields)
- **Graph execution**: Dynamic sequential node dispatch with runtime tensor registry
- **Tensor architecture**: Type-agnostic byte bucket (`std::vector<uint8_t>`) with `Dtype` enum for multi-type support
- **Operators**: Conv2D, ReLU, MaxPool2D, Reshape, Gemm (with transB support)
- **Profiling**: Per-layer chrono timing on all ops + total inference timing
- **Tested on**: [MNIST digit classification](https://github.com/dark-knightshanks/CNN) (~28.9K params, 100% accuracy on test set)

## Profiling Report

Model: 2-layer CNN (Conv→ReLU→MaxPool→Conv→ReLU→MaxPool→Reshape→Gemm)

Build: `g++ -O3 -march=native`

| Op | Avg Latency | % of Total |
|---|---|---|
| Conv2D #1 `[16,1,5,5]` | ~850 µs | ~12% |
| Conv2D #2 `[32,16,5,5]` | ~5,800 µs | **~82%** |
| ReLU (×2) | ~35 µs | <1% |
| MaxPool (×2) | ~30 µs | <1% |
| Reshape | ~3 µs | <1% |
| Gemm `[10,1568]` | ~20 µs | <1% |
| **Total inference** | **~6,900 µs** | |

Conv2D dominates — 7 nested loops, no tiling, no SIMD. Engine is fully compute-bound with 5.23 IPC, 0.02% L1 cache miss rate, and 0.03% branch misprediction.

## Project Structure

```text
blackbox/
├── include/
│   ├── tensor.h          # Tensor class (shape + generic byte buffer + Dtype enum)
│   ├── ops.h             # Op function declarations
│   ├── engine.h          # Graph node struct and engine declarations
│   └── onnx.proto3.pb.h  # Generated Protobuf headers
├── src/
│   ├── engine.cpp        # ONNX model loading, graph parsing, inference dispatch
│   └── ops.cpp           # Op implementations (Conv2D, Gemm, ReLU, MaxPool, Reshape)
├── tests/
│   ├── test_mnist.cpp    # MNIST end-to-end inference and validation
│   └── assets/           # MNIST test images (digit_0.bin – digit_9.bin)
├── docs/
│   ├── runtime-engine.md # Runtime engine function documentation
│   ├── operations.md     # Mathematical operations documentation
│   ├── tensor.md         # Tensor architecture and quantization structs
│   └── tests.md          # Testing application documentation
├── Makefile
├── CONTRIBUTING.md
└── README.md
```

## Future Work

- [x] **Tensor redesign** — generic byte bucket with multi-dtype support (FP32/FP16/INT8) and quantization block structs (Q8_0, Q4_0)
- [x] **Library restructure** — separated core engine from test applications, established contributing guidelines
- [ ] **More CNN ops** — BatchNorm, AvgPool, GlobalAvgPool, Concat, Add (with broadcasting), Pad
- [ ] **Transformer ops** — LayerNorm, RMSNorm, Softmax, GELU/SiLU, Transpose, MatMul (batched)
- [ ] **Memory-mapped weights** — `mmap` for ONNX external data files for zero-copy, instant model loading
- [ ] **INT quantization** — Q8_0 and Q4_0 block quantization, quantized dot product (int8×int8 → int32 accumulate)
- [ ] **NHWC layout** — rewrite Conv2D/Pool for channels-last memory order, benchmark cache improvement vs NCHW
- [ ] **Backend abstraction** — pluggable backends for CPU (scalar), AVX2/NEON (SIMD), CUDA (GPU)
