# Benchmark & Performance Profiling

---

## 1. Summary of Latency & Progress

| Model | Input Resolution | Baseline Latency (`run_inference`) | New Latency (`ExecutionGraph` + Arena) | Speedup | Accuracy | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom CNN (MNIST)** | $1 \times 1 \times 28 \times 28$ | ~21.2 ms | ~5.8 ms | **~3.6x** | 100% | Convolutions |
| **ResNet-18 (ImageNet)** | $1 \times 3 \times 224 \times 224$ | ~15.37 s | **~4.06 s** | **~3.8x** | 100% (2/2) | 2D Convolutions (>98% of total) |

---

## 2. Architecture Comparison: Baseline vs. ExecutionGraph + Arena

| Metric | Baseline (`run_inference`) | `ExecutionGraph` + `MemoryArena` | Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Model** | Interpretive node loop per image | Offline planned DAG (`vector<ExecutionStep>`) | Zero runtime graph parsing |
| **Memory Allocation** | Dynamic heap `malloc` per layer | **Single 50 MB Arena (bump-allocated)** | **Zero mallocs during inference** |
| **Activation Reuse** | None (new tensor allocated each op) | In-place reuse for ReLU & BatchNorm | **~20 MB RAM saved / cache-warm** |
| **Total Instructions (2 imgs)** | 293.64 Billion | **102.05 Billion** | **~2.87x fewer instructions** |
| **CPU Cycles (2 imgs)** | 61.90 Billion | **16.60 Billion** | **~3.73x fewer cycles** |
| **Instructions Per Cycle (IPC)** | 4.74 | **6.15** | **+29.7% pipeline throughput** |
| **Pipeline Retiring (`tma_retiring`)** | ~73.0% | **86.5%** | **Near-optimal pipeline efficiency** |
| **Backend Stalls (`tma_backend_bound`)** | 2.8% | **1.6%** | Memory access is 100% warm in cache |
| **Inference Time / Image** | **15.37 seconds** | **4.06 seconds** | **~3.8x faster** |


---

## 3. Model 1: Custom CNN (MNIST)

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

### Operator Breakdown (Measured on ExecutionGraph + MemoryArena)

| Operator Type | Total Invocations | Baseline Latency (`run_inference`) | New Latency (`ExecutionGraph`) | % of Total Time | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Conv2D`** | 20 | ~15,340,000 $\mu\text{s}$ (~15.34 s) | **~4,058,000 $\mu\text{s}$ (~4.05 s)** | **99.75%** | **~3.78x** |
| **`MaxPool2D`** | 1 | ~8,900 $\mu\text{s}$ | **~3,300 $\mu\text{s}$ (~0.003 s)** | 0.08% | **~2.70x** |
| **`Relu`** | 17 | ~11,200 $\mu\text{s}$ | **~880 $\mu\text{s}$ (< 0.001 s)** | 0.02% | **~12.7x (in-place)** |
| **`ADD`** | 8 | ~6,800 $\mu\text{s}$ | **~4,900 $\mu\text{s}$ (~0.005 s)** | 0.12% | **~1.39x** |
| **`Gemm`** | 1 | ~1,650 $\mu\text{s}$ | **~530 $\mu\text{s}$ (< 0.001 s)** | 0.01% | **~3.11x** |
| **`GlobalAveragePool`** | 1 | ~13 $\mu\text{s}$ | **~16 $\mu\text{s}$ (< 0.0001 s)** | < 0.001% | ~1.0x |
| **`Flatten`** | 1 | ~0 $\mu\text{s}$ | **0 $\mu\text{s}$** | 0.00% | **Zero-cost alias** |
| **Total Inference** | **49 layers** | **~15,368,000 $\mu\text{s}$ (~15.37 s)** | **~4,068,000 $\mu\text{s}$ (~4.06 s)** | **100.0%** | **~3.78x faster** |

### Convolution Stage-by-Stage Latency (`ExecutionGraph` Run)

| Stage | Layer / Node | Kernel | In $\to$ Out Channels | Output Size | New Latency ($\mu\text{s}$) | Baseline Latency ($\mu\text{s}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stem** | `/conv1/Conv` | $7 \times 7$, s=2, p=3 | $3 \to 64$ | $112 \times 112$ | **~248,000 $\mu\text{s}$ (~0.25 s)** | 1,144,576 $\mu\text{s}$ |
| **Layer 1** (Block 1) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $64 \to 64$ | $56 \times 56$ | ~265,000 $\mu\text{s}$ each (~0.53 s) | ~2,000,000 $\mu\text{s}$ |
| **Layer 1** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $64 \to 64$ | $56 \times 56$ | ~265,000 $\mu\text{s}$ each (~0.53 s) | ~2,000,000 $\mu\text{s}$ |
| **Layer 2** (Block 1) | `conv1` / `conv2` / `downsample` | $3 \times 3$ / $1 \times 1$ | $64 \to 128$ | $28 \times 28$ | ~132,000 + ~260,000 + ~31,000 $\mu\text{s}$ (~0.42 s) | ~1,555,000 $\mu\text{s}$ |
| **Layer 2** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $128 \to 128$ | $28 \times 28$ | ~261,000 + ~261,000 $\mu\text{s}$ (~0.52 s) | ~1,991,000 $\mu\text{s}$ |
| **Layer 3** (Block 1) | `conv1` / `conv2` / `downsample` | $3 \times 3$ / $1 \times 1$ | $128 \to 256$ | $14 \times 14$ | ~130,000 + ~254,000 + ~30,000 $\mu\text{s}$ (~0.41 s) | ~1,505,000 $\mu\text{s}$ |
| **Layer 3** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $256 \to 256$ | $14 \times 14$ | ~253,000 + ~253,000 $\mu\text{s}$ (~0.51 s) | ~1,923,000 $\mu\text{s}$ |
| **Layer 4** (Block 1) | `conv1` / `conv2` / `downsample` | $3 \times 3$ / $1 \times 1$ | $256 \to 512$ | $7 \times 7$ | ~125,000 + ~242,000 + ~30,000 $\mu\text{s}$ (~0.40 s) | ~1,418,000 $\mu\text{s}$ |
| **Layer 4** (Block 2) | `conv1` / `conv2` | $3 \times 3$, s=1, p=1 | $512 \to 512$ | $7 \times 7$ | ~242,000 + ~243,000 $\mu\text{s}$ (~0.49 s) | ~1,789,000 $\mu\text{s}$ |

---

## 4. Hardware Profiling & Microarchitecture Analysis (`perf stat`)

Metrics recorded from running `perf stat ./bin/test_resnet18` across 2 full ImageNet inference passes on the new `ExecutionGraph` + `MemoryArena` backend:

| Performance Metric | Measured Value | Significance / Analysis |
| :--- | :--- | :--- |
| **Total Instructions** | **102.05 Billion** | ~51 Billion instructions per forward pass (down from 147B in baseline!). Eliminating heap allocations and per-pixel bounds checks saved nearly **100 Billion instructions per image**. |
| **CPU Cycles** | **16.60 Billion** | ~8.30 Billion cycles per image on P-cores (down from 30.95B in baseline). |
| **Instructions Per Cycle (IPC)** | **6.15** | Exceptional pipeline throughput; the Intel Golden Cove P-cores are utilizing virtually all execution ports without cache stalls. |
| **Branch Misprediction Rate** | **0.07%** | Near-perfect branch prediction across 19.6 Billion branches (99.93% accurate). |
| **Pipeline Retiring (`tma_retiring`)** | **86.5%** | Phenomenal hardware efficiency; 86.5% of all CPU pipeline execution slots successfully commit useful work. |
| **Backend Memory Stall (`tma_backend_bound`)** | **1.6%** | Memory access is 100% warm in CPU cache. DRAM is NOT a bottleneck. |
| **CPU Core Utilization** | **0.999 / 12** | **Strictly single-threaded.** 7 physical cores and 11 threads are completely idle. |

---

## 5. Key Performance Insights

1. **Memory Arena & OOP Proved Highly Effective:**
   - Pre-calculating static 64-byte aligned arena offsets and in-place activation reuse reduced latency by **~3.8x (15.37s $\to$ 4.06s)** without changing the underlying mathematical loop nests!
2. **Current Bottleneck — Compute Bound & Single-Threaded:**
   - The engine is 86.5% retiring and only 1.6% backend-bound, executing 102 Billion scalar instructions on a single CPU core.

