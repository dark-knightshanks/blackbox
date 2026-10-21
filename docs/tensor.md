# Tensor Architecture

The `include/tensor.h` file defines the core data structure used to hold multi-dimensional tensor data throughout the inference engine.

Instead of typing the tensor to a single fixed numeric format, the architecture uses a generic byte bucket alongside a metadata flag and non-owning pointer views for zero-copy memory mapping and arena allocation.

---

### Core Container

```cpp
class Tensor {
public:
    Dtype flag = FP32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;               // Owning storage for standalone buffers
    const uint8_t* external_ptr = nullptr;   // Non-owning view for mmap weights & Arena buffers

    size_t size() const;
    size_t byte_size() const;
    const uint8_t* raw_data() const;
};
```
***Tensor*** - The primary data container:
*   **`shape`**: Tracks the multi-dimensional layout (e.g., `[Batch, Channels, Height, Width]`).
*   **`data`**: Flat, 1D array of `uint8_t` bytes acting as heap-allocated memory for standalone buffers.
*   **`external_ptr`**: Non-owning pointer pointing directly into memory-mapped files (`mmap`), microcontroller Flash (`.rodata`), or the `MemoryArena`, avoiding duplicate RAM allocations.
*   **`raw_data()`**: Accessor that returns `external_ptr` if mapped/allocated in the arena, or `data.data()` if owning.
*   **`byte_size()`**: Helper function that calculates required memory allocation based on `shape` and `Dtype`.

---

### Integration with Memory Arena

In the zero-allocation `ExecutionGraph` pipeline:
* **Intermediate Activations:** Do not use `tensor.data.resize()`. Instead, `external_ptr` points directly to an offset within the pre-allocated `MemoryArena`.
* **Weights:** Point to the memory-mapped `.onnx.data` file via `external_ptr`.
* Result: **Zero dynamic heap allocations** during inference passes.

---

### Data Types & Quantization

The engine supports multi-precision formats via the `Dtype` enum and custom block structs:

```cpp
enum Dtype { FP32, FP16, I32, I8, Q4_0, Q8_0 };
```

```cpp
struct blockQ8_0 {
    uint16_t scale;
    int8_t arr[32];     // 32 quantized 8-bit integers in a block
};
```
***blockQ8_0*** - A quantization block that compresses 32 floats into 8-bit signed integers with a shared 16-bit float scale factor, significantly reducing memory bandwidth.

```cpp
struct blockQ4_0 {
    uint16_t scale;
    uint8_t arr[16];    // 32 quantized 4-bit integers (2 per byte)
};
```
***blockQ4_0*** - A 4-bit quantization block packing two 4-bit nibbles per byte for extreme 8x memory compression.
