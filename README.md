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
- **Zero-copy memory mapping**: POSIX `mmap()` for ONNX external weights with buffer caching (`get_mmap_buffer`) for instant model loading with 0 bytes RAM duplication
- **Graph execution**: Dynamic sequential node dispatch with runtime tensor registry
- **Tensor architecture**: Type-agnostic byte bucket (`std::vector<uint8_t>`) + non-owning `external_ptr` view and `Dtype` enum
- **Modular Operators**:
  - **Conv / Pooling**: Conv2D (4D), MaxPool2D
  - **Activations**: ReLU, GELU, SiLU, Softmax (numerically stable 3-pass)
  - **Normalization**: LayerNorm, RMSNorm
  - **Linear / MatMul**: Gemm (with transB), Batched 4D MatMul (with broadcasting)
  - **Shape**: Reshape (with `-1` inference), Transpose (4D permutations), Argmax
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
| Reshape | ~0–1 µs | <1% (Zero-copy view) |
| Gemm `[10,1568]` | ~20 µs | <1% |
| **Total inference** | **~6,700 µs** | |

Conv2D dominates — 7 nested loops, no tiling, no SIMD. Engine is fully compute-bound with 5.23 IPC, 0.02% L1 cache miss rate, and 0.03% branch misprediction.

## Project Structure

```text
blackbox/
├── include/
│   ├── tensor.h             # Tensor container (shape + byte buffer + external_ptr + Dtype enum)
│   ├── ops.h                # Operator function declarations
│   ├── engine.h             # Graph node struct and engine declarations
│   └── onnx.proto3.pb.h     # Generated Protobuf headers
├── src/
│   ├── engine.cpp           # ONNX loading, zero-copy mmap cache, graph inference dispatch
│   ├── ops_activations.cpp  # ReLU, GELU, SilU, Softmax
│   ├── ops_conv.cpp         # Conv2D, MaxPool2D
│   ├── ops_matmul.cpp       # Gemm, 4D Batched MatMul
│   ├── ops_norm.cpp         # LayerNorm, RMSNorm
│   └── ops_shape.cpp        # Reshape, Transpose, Argmax
├── tests/
│   ├── test_mnist.cpp       # MNIST end-to-end inference and validation
│   └── assets/              # MNIST test images (digit_0.bin – digit_9.bin)
├── docs/
│   ├── runtime-engine.md    # Runtime engine function documentation
│   ├── operations.md        # Mathematical operations documentation
│   ├── tensor.md            # Tensor architecture and quantization structs
│   └── tests.md             # Testing application documentation
├── Makefile
├── CONTRIBUTING.md
└── README.md
```

## Future Work

- [x] **Tensor redesign** — generic byte bucket with multi-dtype support (FP32/FP16/INT8) and quantization block structs (Q8_0, Q4_0)
- [x] **Library restructure** — modularized operator files, separated core engine from test applications, established contributing guidelines
- [x] **Memory-mapped weights** — POSIX `mmap()` for ONNX external data files for zero-copy, instant model loading with file caching
- [x] **Transformer ops** — LayerNorm, RMSNorm, Softmax, GELU/SiLU, 4D Transpose, Batched MatMul
- [ ] **CNN ops (In Progress)** — BatchNorm, AvgPool, GlobalAvgPool, Concat, Add (with broadcasting), Pad
- [ ] **INT quantization** — Q8_0 and Q4_0 block quantization, quantized dot product (int8×int8 → int32 accumulate)
- [ ] **NHWC layout** — rewrite Conv2D/Pool for channels-last memory order, benchmark cache improvement vs NCHW
- [ ] **Backend abstraction** — pluggable backends for CPU (scalar), AVX2/NEON (SIMD), CUDA (GPU)
