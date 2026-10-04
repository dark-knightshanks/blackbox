# Runtime Engine

<<<<<<< Updated upstream
<<<<<<< HEAD
The `src/engine.cpp` file contains the core runtime engine functions for parsing ONNX models, memory-mapping external weights, and executing the computational graph.

### Core Functions & State:

```cpp
const uint8_t* get_mmap_buffer(const std::string& filepath);
```
* ***get_mmap_buffer*** - Opens the external binary weight file (e.g. `model.onnx.data`), queries its size with `fstat`, and maps it directly into the process virtual address space via POSIX `mmap()` (`PROT_READ`, `MAP_SHARED`). Caches open file pointers in a static hash map (`mmap_cache`) so each binary file is mapped exactly once with zero heap RAM duplication.

```cpp
void loadinitializer(const onnx::GraphProto& graph);
```
* ***loadinitializer*** - Loads model weights from `.onnx` and memory-mapped external files into the global `weight` registry. If data is stored externally, it assigns a non-owning pointer `tensor.external_ptr = base + offset` directly into the mapped virtual address space.

```cpp
std::unordered_map<std::string, Tensor> weight;
```
* Note: The `loadinitializer` function stores all parsed weights into this global hash map, allowing `run_inference` to quickly look them up by their string names.

```cpp
Tensor run_inference(const onnx::GraphProto& graph, const Tensor& input_image);
```
* ***run_inference*** - Traverses each node in the computational graph in topological order and dynamically dispatches to the corresponding mathematical operation functions (like `run_conv2D`, `run_gemm`, `run_relu`, `run_reshape`).
=======
The engine.cpp contains the main core runtime engine funnctions for parsing the model and executing the computational graph
=======
The `src/engine.cpp` file contains the core runtime engine functions for parsing ONNX models, memory-mapping external weights, and executing the computational graph.
>>>>>>> Stashed changes

### Core Functions & State:

```cpp
const uint8_t* get_mmap_buffer(const std::string& filepath);
```
* ***get_mmap_buffer*** - Opens the external binary weight file (e.g. `model.onnx.data`), queries its size with `fstat`, and maps it directly into the process virtual address space via POSIX `mmap()` (`PROT_READ`, `MAP_SHARED`). Caches open file pointers in a static hash map (`mmap_cache`) so each binary file is mapped exactly once with zero heap RAM duplication.

```cpp
void loadinitializer(const onnx::GraphProto& graph);
```
* ***loadinitializer*** - Loads model weights from `.onnx` and memory-mapped external files into the global `weight` registry. If data is stored externally, it assigns a non-owning pointer `tensor.external_ptr = base + offset` directly into the mapped virtual address space.

```cpp
std::unordered_map<std::string, Tensor> weight;
```
* Note: The `loadinitializer` function stores all parsed weights into this global hash map, allowing `run_inference` to quickly look them up by their string names.

```cpp
Tensor run_inference(const onnx::GraphProto& graph, const Tensor& input_image);
```
<<<<<<< Updated upstream
*  ***run_infernce*** -  Traverses each node in the computational graph and dynamically dispatches to the corresponding mathematical operation functions (like `run_conv2D` or `run_gemm`).
>>>>>>> origin/main
=======
* ***run_inference*** - Traverses each node in the computational graph in topological order and dynamically dispatches to the corresponding mathematical operation functions (like `run_conv2D`, `run_gemm`, `run_relu`, `run_reshape`).
>>>>>>> Stashed changes
