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

Then build and run the test suites:
```bash
make bin/test_mnist      # builds MNIST CNN test
./bin/test_mnist

make bin/test_resnet18   # builds ResNet-18 ImageNet test
./bin/test_resnet18

make clean               # removes build artifacts
```

Expects `model.onnx` + `model.onnx.data` (MNIST) or `resnet18.onnx` + `resnet18.onnx.data` (ImageNet) in project root and test tensors in `tests/assets/`.

## Features

- **Model loading**: ONNX format via protobuf (raw data, external data, float/int64 fields)
- **Zero-copy memory mapping**: POSIX `mmap()` for ONNX external weights with buffer caching (`get_mmap_buffer`) for instant model loading with 0 bytes RAM duplication
- **Graph execution**: Dynamic sequential node dispatch with runtime tensor registry
- **Tensor architecture**: Type-agnostic byte bucket (`std::vector<uint8_t>`) + non-owning `external_ptr` view and `Dtype` enum
- **Modular Operators**:
  - **Conv / Pooling**: Conv2D (4D), MaxPool2D, GlobalAveragePool
  - **Activations**: ReLU, GELU, SiLU, Softmax (numerically stable 3-pass)
  - **Normalization**: LayerNorm, RMSNorm, BatchNormalization
  - **Linear / MatMul**: Gemm (with transB), Batched 4D MatMul (with broadcasting)
  - **Shape / Residual**: Reshape (with `-1` inference), Transpose (4D permutations), Flatten, Add (broadcasting), Concat, Pad, Argmax
- **Profiling**: Per-layer chrono timing on all ops + total inference timing and microarchitectural hardware analysis
- **Tested Models**:
  - [MNIST digit classification](https://github.com/dark-knightshanks/CNN) (~28.9K params, **100% accuracy** on test set)
  - **ResNet-18 (ImageNet)** (~11.7M params, **100% accuracy** on ImageNet test samples)

## Performance & Benchmarks

Full per-layer profiling, operator breakdowns, and microarchitectural hardware counters (`perf stat`) are documented in [**`benchmark.md`**](benchmark.md).

| Model | Input Resolution | Parameters | Baseline Latency | Accuracy | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom CNN (MNIST)** | $1 \times 1 \times 28 \times 28$ | ~21.8 K | **~21.2 ms** | 100% (10/10) | `node_conv2d_1` (~84.5% of total) |
| **ResNet-18 (ImageNet)** | $1 \times 3 \times 224 \times 224$ | ~11.7 M | **~15.37 s** | 100% (2/2) | 2D Convolutions (~99.8% of total) |

Conv2D dominates — 7 nested loops, no tiling, no SIMD. 

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
│   ├── ops_conv.cpp         # Conv2D, MaxPool2D, GlobalAveragePool
│   ├── ops_matmul.cpp       # Gemm, 4D Batched MatMul
│   ├── ops_norm.cpp         # LayerNorm, RMSNorm, BatchNormalization
│   └── ops_shape.cpp        # Reshape, Transpose, Flatten, Add, Concat, Pad, Argmax
├── tests/
│   ├── test_mnist.cpp       # MNIST end-to-end inference and validation
│   ├── test_resnet18.cpp    # ResNet-18 ImageNet validation
│   ├── test_vit.cpp         # Vision Transformer test harness
│   ├── inspect_vit.cpp      # ONNX model operator inspector
│   └── assets/              # MNIST and ImageNet preprocessed test tensors
├── benchmark.md             # In-depth latency breakdown & perf stat hardware metrics
├── Makefile
├── CONTRIBUTING.md
└── README.md
```

## Future Work

- [x] **Tensor redesign** — generic byte bucket with multi-dtype support (FP32/FP16/INT8) and quantization block structs (Q8_0, Q4_0)
- [x] **Library restructure** — modularized operator files, separated core engine from test applications, established contributing guidelines
- [x] **Memory-mapped weights** — POSIX `mmap()` for ONNX external data files for zero-copy, instant model loading with file caching
- [x] **Transformer ops** — LayerNorm, RMSNorm, Softmax, GELU/SiLU, 4D Transpose, Batched MatMul
- [x] **CNN ops** — BatchNormalization, GlobalAveragePool, Concat, Add (with broadcasting), Pad, Flatten
- [ ] **Vision Transformer (ViT) Support** — dynamic shape operators (`Shape`, `Gather`, `Expand`, `Slice`), elementwise math (`Mul`, `Div`, `Erf`, `Where`, `IsNaN`)
- [ ] **INT quantization** — Q8_0 and Q4_0 block quantization, quantized dot product (int8×int8 → int32 accumulate)
- [ ] **NHWC layout** — rewrite Conv2D/Pool for channels-last memory order, benchmark cache improvement vs NCHW
- [ ] **SIMD & Multithreading** — OpenMP parallelization across CPU cores, AVX2 FMA vectorization
- [ ] **Backend abstraction** — pluggable backends for CPU (scalar), AVX2/NEON (SIMD), CUDA (GPU)
