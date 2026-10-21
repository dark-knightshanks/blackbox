# Runtime Engine

`blackbox` uses a two-stage, zero-malloc execution pipeline inspired by production inference runtimes.

---

## 1. How It Works

1. **Compilation Phase (`build_execution_graph`)**:
   - Parses the ONNX model graph once at startup.
   - Calculates intermediate tensor shapes and assigns fixed byte offsets into a single memory arena.
   - Reuses memory in-place for elementwise operations (`ReLU`, `BatchNorm`).
   - Packages operations into an ordered sequence of steps.

2. **Execution Phase (`ExecutionGraph::run`)**:
   - Copies the input image to offset `0` of the pre-allocated arena.
   - Sequentially invokes each operator's `forward()` method using raw pointers into the arena and memory-mapped weights.
   - Copies the final classification logits out of the arena.
   - **Zero heap allocations (`malloc`/`free`) occur during inference.**

---

## 2. Core Components

### `MemoryArena` (`include/arena.h`)
- A single contiguous memory block allocated with 64-byte alignment via `aligned_alloc` for AVX2/AVX-512 vector operations and cache-line efficiency.
- Hands out memory via fast bump-pointer arithmetic: `(offset + 63) & ~63`.
- Resets in $O(1)$ between inference passes without deallocating.
- Supports move semantics (`&&`) for zero-copy ownership transfer.

### `Operator` (`include/ops.h`)
- Abstract base class for all compute kernels:
  ```cpp
  virtual void forward(uint8_t* arena,
                       const std::vector<size_t>& in_offsets,
                       const std::vector<const uint8_t*>& weight_ptrs,
                       const std::vector<size_t>& out_offsets) = 0;
  ```
- Separates memory management from mathematical compute. Activations reside in the arena; static weights reside in zero-copy memory-mapped storage.

### `ExecutionGraph` (`include/execution.h`)
- Holds the pre-allocated `MemoryArena` and the planned `ExecutionStep` list.
- Contains the generic forward loop and real-time microsecond-level kernel profiling via `std::chrono`.

### Memory-Mapped Weights (`src/engine.cpp`)
- Maps external `.onnx.data` binary weight files directly into process virtual memory using POSIX `mmap()` (`PROT_READ`, `MAP_SHARED`).
- Avoids duplicating model weights in heap RAM.
