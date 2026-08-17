# blackbox

A from-scratch ML inference engine written in C++17 — built to understand every layer of the stack, from model parsing to quantized compute.

Currently runs ONNX CNN models on CPU with FP32 precision. Being developed into a general-purpose inference library with quantization, layout optimization, and multi-backend support.

## Building

**Prerequisites**: `g++` (C++17), `libprotobuf-dev`

```bash
make            # builds bin/onnx_engine
./bin/onnx_engine
make clean      # removes build artifacts
```

Expects `model.onnx` + `model.onnx.data` in project root and test images in `assets/`.

## What's Done

- **Model loading**: ONNX format via protobuf (raw data, external data, float/int64 fields)
- **Graph execution**: Sequential node dispatch with runtime tensor registry
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

Conv2D dominates — 7 nested loops, no tiling, no SIMD.

## Project Structure

```
blackbox/
├── include/
│   ├── tensor.h          # Tensor class (shape + flat FP32 buffer)
│   ├── ops.h             # Op function declarations
│   └── include.h         # Graph node struct
├── src/
│   ├── main.cpp          # Model loading, graph execution, inference loop
│   └── ops.cpp           # Op implementations with per-op profiling
├── assets/               # MNIST test images (digit_0.bin – digit_9.bin)
├── Makefile
└── README.md
```

## Future Work

- [ ] **Tensor redesign** — multi-dtype support (FP32/FP16/INT8), layout enum (NCHW/NHWC), ownership model (owned/view/mmap)
- [ ] **More CNN ops** — BatchNorm, AvgPool, GlobalAvgPool, Concat, Add/Mul/Sub/Div with broadcasting, Transpose, Sigmoid, Tanh
- [ ] **NHWC layout** — rewrite Conv2D/Pool for channels-last memory order, benchmark cache improvement vs NCHW
- [ ] **INT quantization** — Q8_0 and Q4_0 block quantization, quantized dot product (int8×int8 → int32 accumulate)
- [ ] **GGUF format** — parser for loading pre-quantized models via mmap, ONNX → GGUF converter
- [ ] **Backend abstraction** — pluggable backends for CPU (scalar), ARM NEON (SIMD), CUDA (GPU)
- [ ] **Transformer ops** — LayerNorm, Softmax, GELU, multi-head attention (ViT support)

