# Test Suite & Verification

The `tests/` directory contains unit tests, model verification suites, and architecture inspectors.

---

## Available Test Targets

| Binary Target | Source File | Description | Verification Status |
| :--- | :--- | :--- | :--- |
| **`bin/test_resnet18`** | `tests/test_resnet18.cpp` | End-to-end ImageNet classification on ResNet-18 using `build_execution_graph` and `MemoryArena`. | **100% Accuracy (2/2)** (Samoyed: Class 258, Egyptian Cat: Class 285) |
| **`bin/test_mnist`** | `tests/test_mnist.cpp` | End-to-end handwritten digit classification on 10 MNIST test samples. | **100% Accuracy (10/10)** |
| **`bin/test_arena`** | `tests/test_arena.cpp` | Standalone unit test suite for `MemoryArena` (8 test cases: 64-byte alignment, non-overlapping chunks, float read/write, reset, and overflow detection). | **8/8 PASS** |
| **`bin/test_vit`** | `tests/test_vit.cpp` | End-to-end verification harness for Vision Transformer (`vit-base-patch16-224.onnx`). | Under active development |

---

## How to Build & Run Tests

```bash
# Build and run the MemoryArena unit test
make bin/test_arena
./bin/test_arena

# Build and run the ResNet-18 verification
make bin/test_resnet18
./bin/test_resnet18

# Build and run MNIST test
make bin/test_mnist
./bin/test_mnist
```
