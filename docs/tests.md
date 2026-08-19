# Tests

The `tests/` directory contains standalone applications and scripts used to validate the inference engine against specific models and datasets.

### Current Tests

*   **`test_mnist.cpp`**: Runs a full end-to-end inference test on a 2-layer CNN using 10 raw binary images from the MNIST dataset (located in `tests/assets/`). It sequentially loads each image, runs it through the engine, and validates that the engine's `argmax` classification perfectly matches the expected digit.
