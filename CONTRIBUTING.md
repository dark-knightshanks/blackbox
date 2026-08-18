# Contributing to blackbox

## Commit Messages

Write commit messages with a title and description:

```
<area>: <short summary>

<detailed description of what and why>

Signed-off-by: Your Name <your@email.com>
```

### Example

```
ops: updated functions using tensor to use byte-level pointer access 

Updates all mathematical operations(conv2d, maxpool2d, gemm, argmax, relu,
parse_target_shape)functions as well data and image loader functions(load_image,
loadinitiallizer) to use the new uint8_t Tensor data container.Implements reinterpret_cast for safe memory access and transitions bounds checking to calculate logical element sizes instead of raw byte counts.

Signed-off-by: Rishiraj Rajgor <rishi@example.com>
```

### Areas

Use one of these prefixes in the title for the area:

- `tensor` - core tensor struct, memory layouts, data types
- `ops` - mathematical operator implementations (Conv2D, Gemm, etc.)
- `engine` - runtime logic, graph parsing, node dispatching
- `loader` - reading/loading ONNX or GGUF files and weight buffers
- `build` - Makefile, toolchain configuration
- `tests` - test scripts or asset validation
- `docs` - documentation and README updates

## Branches

When contributing, create a properly named branch to be merged.

- Feature branches - `<area>/<feature>`, e.g. `tensor/byte-bucket` or `ops/quantized-gemm`

## Build and Test

Before pushing, make sure your code compiles and runs successfully:

```bash
make clean && make
./bin/onnx_engine
```
