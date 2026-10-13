# Benchmark & Performance Profiling

---

## 1. Summary of Baseline Latency

| Model | Input Resolution | Parameters | Baseline Latency | Accuracy | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom CNN (MNIST)** | $1 \times 1 \times 28 \times 28$ | ~21.8 K | **~21.2 ms** | 100% (10/10) | `node_conv2d_1` (~84.5% of total) |
| **ResNet-18 (ImageNet)** | $1 \times 3 \times 224 \times 224$ | ~11.7 M | **~15.37 s** | 100% (2/2) | 2D Convolutions (~99.8% of total) |

---

## 2. Model 1: Custom CNN (MNIST)

- **Input:** Single grayscale digit image ($1 \times 1 \times 28 \times 28$)
- **Architecture:** `Conv2D` $\to$ `ReLU` $\to$ `MaxPool2D` $\to$ `Conv2D` $\to$ `ReLU` $\to$ `MaxPool2D` $\to$ `Reshape` $\to$ `Linear (Gemm)`

### Per-Layer Latency Profile (Averaged over 10 test runs)

| Node Name | Operator | Input Shape | Kernel / Stride | Output Shape | Avg Latency ($\mu\text{s}$) | % Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `node_conv2d` | `Conv` | `[1, 1, 28, 28]` | $5 \times 5$, pad 0, stride 1 | `[1, 16, 24, 24]` | ~2,550 $\mu\text{s}$ | 12.0% |
| `node_relu` | `Relu` | `[1, 16, 24, 24]` | elementwise | `[1, 16, 24, 24]` | ~35 $\mu\text{s}$ | 0.2% |
| `node_max_pool2d` | `MaxPool` | `[1, 16, 24, 24]` | $2 \times 2$, stride 2 | `[1, 16, 12, 12]` | ~80 $\mu\text{s}$ | 0.4% |
| `node_conv2d_1` | `Conv` | `[1, 16, 12, 12]` | $5 \times 5$, pad 0, stride 1 | `[1, 32, 8, 8]` | **~18,150 $\mu\text{s}$** | **85.5%** |
| `node_relu_1` | `Relu` | `[1, 32, 8, 8]` | elementwise | `[1, 32, 8, 8]` | ~18 $\mu\text{s}$ | < 0.1% |
| `node_max_pool2d_1` | `MaxPool` | `[1, 32, 8, 8]` | $2 \times 2$, stride 2 | `[1, 32, 4, 4]` | ~39 $\mu\text{s}$ | 0.2% |
| `node_view` | `Reshape` | `[1, 32, 4, 4]` | reshape to `[1, 512]` | `[1, 512]` | ~5 $\mu\text{s}$ | < 0.1% |
| `node_linear` | `Gemm` | `[1, 512]` | $512 \times 10$ | `[1, 10]` | ~52 $\mu\text{s}$ | 0.2% |
| **Total Graph Run** | | | | | **~21,220 $\mu\text{s}$ (~21.2 ms)** | **100%** |

---

## 3. Model 2: ResNet-18 (ImageNet)

- **Input:** RGB color image ($1 \times 3 \times 224 \times 224$)
- **Architecture:** 7x7 Conv + MaxPool, followed by 4 Residual stages (8 residual blocks = 16 convolutions + 3 downsample convolutions), Global Average Pooling, Flatten, and Linear (Gemm 1000 classes).

### Operator Breakdown

| Operator Type | Total Invocations | Aggregated Latency ($\mu\text{s}$) | Aggregated Latency (s) | % of Total Time |
| :--- | :--- | :--- | :--- | :--- |
| **`Conv`** | 20 | ~15,340,000 $\mu\text{s}$ | **~15.34 s** | **99.87%** |
| **`MaxPool`** | 1 | ~8,900 $\mu\text{s}$ | ~0.009 s | 0.06% |
| **`Relu`** | 17 | ~11,200 $\mu\text{s}$ | ~0.011 s | 0.07% |
| **`Add`** | 8 | ~6,800 $\mu\text{s}$ | ~0.007 s | 0.04% |
| **`Gemm`** | 1 | ~1,650 $\mu\text{s}$ | ~0.002 s | 0.01% |
| **`GlobalAveragePool`** | 1 | ~13 $\mu\text{s}$ | < 0.0001 s | < 0.001% |
| **`Flatten`** | 1 | ~0 $\mu\text{s}$ | 0 s | 0.00% (zero-copy) |
| **Total Inference** | **49 layers** | **~15,368,000 $\mu\text{s}$** | **~15.37 s** | **100.0%** |

### Convolution Stage-by-Stage Latency

| Stage | Layer / Node | Kernel | In $\to$ Out Channels | Output Size | Latency ($\mu\text{s}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stem** | `/conv1/Conv` | $7 \times 7$, s=2, p=3 | $3 \to 64$ | $112 \times 112$ | **1,144,576 $\mu\text{s}$ (~1.14 s)** |
| **Layer 1** (Block 1) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $64 \to 64$ | $56 \times 56$ | ~1,000,000 $\mu\text{s}$ each (~2.00 s) |
| **Layer 1** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $64 \to 64$ | $56 \times 56$ | ~1,000,000 $\mu\text{s}$ each (~2.00 s) |
| **Layer 2** (Block 1) | `conv1` / `conv2` / `downsample` | $3 \times 3$ / $1 \times 1$ | $64 \to 128$ | $28 \times 28$ | ~507,000 + ~990,000 + ~58,000 $\mu\text{s}$ (~1.55 s) |
| **Layer 2** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $128 \to 128$ | $28 \times 28$ | ~993,000 + ~998,000 $\mu\text{s}$ (~1.99 s) |
| **Layer 3** (Block 1) | `conv1` / `conv2` / `downsample` | $3 \times 3$ / $1 \times 1$ | $128 \to 256$ | $14 \times 14$ | ~492,000 + ~957,000 + ~56,000 $\mu\text{s}$ (~1.51 s) |
| **Layer 3** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $256 \to 256$ | $14 \times 14$ | ~974,000 + ~949,000 $\mu\text{s}$ (~1.92 s) |
| **Layer 4** (Block 1) | `conv1` / `conv2` / `downsample` | $3 \times 3$ / $1 \times 1$ | $256 \to 512$ | $7 \times 7$ | ~469,000 + ~893,000 + ~56,000 $\mu\text{s}$ (~1.42 s) |
| **Layer 4** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $512 \to 512$ | $7 \times 7$ | ~897,000 + ~892,000 $\mu\text{s}$ (~1.79 s) |

---

## 4. Hardware Profiling & Microarchitecture Analysis (`perf stat`)

Metrics recorded from running `perf stat ./bin/test_resnet18` across 2 full ImageNet inference passes:

| Performance Metric | Measured Value | Significance / Analysis |
| :--- | :--- | :--- |
| **Total Instructions** | **293.64 Billion** | ~147 Billion instructions per forward pass. Compared against the theoretical **~1.8 Billion FLOPs** of ResNet-18, there is an **~80x instruction overhead** due to scalar 6D loop pointer/index calculations. |
| **CPU Cycles** | 61.90 Billion | ~30.95 Billion cycles per inference pass. |
| **Instructions Per Cycle (IPC)** | **4.74** | Extremely high pipeline efficiency; out-of-order execution engine is retiring almost 5 instructions per cycle without pipeline stalls. |
| **Branch Misprediction Rate** | **0.07%** | Predictable nested loop trip counts result in near-zero branch recovery penalties. |
| **Memory Stall (`tma_backend_bound`)** | **2.8%** | Very low backend stalls indicate the execution is **compute/instruction bound**, not stalled waiting on RAM memory loads. |
| **CPU Core Utilization** | **1.000 / 12** | **Single-threaded execution only.** All remaining CPU threads sit idle throughout inference. |

---

## 5. Key Performance Insights & Next Optimization Steps

1. **Convolution Dominates Entire Engine:**
   - Both models prove that non-convolution operators (Activations, Pooling, Adds, Linear GEMM) represent `< 0.2%` of execution time.
   - Any engine optimization must target `src/ops_conv.cpp`.
2. **Current Bottleneck — Scalar Instruction Bloat:**
   - The direct 6D nested loop performs individual scalar address calculations for every multiply-add, retiring ~147B instructions for 1.8B FLOPs.
   - It runs strictly on 1 core without SIMD / vector instruction registers (AVX2/FMA).
3. **Targets for Next Phase:**
   - **OpenMP Multithreading:** Parallelize outer output channel loops across all CPU threads (immediate near-linear speedup).
   - **`im2col` + GEMM / AVX2 FMA Vectorization:** Pack input patches into contiguous memory and use 256-bit FMA (`_mm256_fmadd_ps`) to process 8 float multiply-accumulates in a single instruction, drastically reducing the 147B instruction count.

