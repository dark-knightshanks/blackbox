# Tensor Architecture

The `tensor.h` file defines the core data structure used to hold multi-dimensional data throughout the inference engine. 

Instead of typing the tensor to a specific numeric format (like `float`), the architecture uses a generic byte bucket alongside a metadata flag to allow for flexible quantization.

### Core Container

```cpp
class Tensor {
public:
    Dtype flag = FP32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;

    size_t size() const;
    size_t byte_size() const;
};
```
***Tensor*** - The primary data container. 
*   **`shape`**: Tracks the multi-dimensional layout (e.g., `[Batch, Channels, Height, Width]`).
*   **`data`**: A flat, 1D array of `uint8_t` bytes acting as generic memory storage.
*   **`byte_size()`**: Helper function that safely calculates required memory allocation based on the current data type.

### Data Types & Quantization

The engine currently is preparing to support multiple data types using the `Dtype` enum and custom block structures:

```cpp
enum Dtype { FP32, FP16, Q8_0, Q4_0 };
```

```cpp
struct blockQ8_0 {
    float d;           // Scaling factor for the block
    int8_t qs[32];     // 32 quantized 8-bit integers
};
```
***blockQ8_0*** - A quantization block that compresses 32 32-bit floats into 8-bit integers with a shared scaling factor `d`, significantly reducing memory usage and bandwidth.

```cpp
struct blockQ4_0 {
    float d;
    uint8_t qs[16];    // 32 quantized 4-bit integers (2 per byte)
};
```
***blockQ4_0*** - A heavier quantization block compressing 32 floats into 4-bit integers for extreme memory savings.
