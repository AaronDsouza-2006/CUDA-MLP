# CUDA-MLP

A **from-scratch** implementation of a Multi-Layer Perceptron (MLP) neural network using **CUDA/C++** for high-performance GPU training. 

This project demonstrates efficient custom CUDA kernels for matrix operations, forward & backward propagation, and is designed to be **general-purpose** — MNIST classification is provided only as an example.

## Features

- Fully CUDA-accelerated forward and backward passes
- Tiled matrix multiplication with shared memory for performance
- ReLU activation + Softmax + Cross-Entropy loss
- Mini-batch Stochastic Gradient Descent (SGD)
- Unified Memory with prefetching
- Model checkpointing
- CPU reference implementation for validation


## Project Structure

```
CUDA-MLP/
├── MLP.cu              # Core MLP class (CUDA kernels)
├── MLP.cuh             # MLP class header
├── matmul.cu           # Custom tiled matrix multiplication kernels
├── matmul.cuh          # Matrix mul headers
├── data_loader.cpp     # MNIST CSV loader (example only)
├── data_loader.hpp
├── MNIST_example.cu    # Example: MNIST classification
├── MNIST_example_cpu.cpp # CPU version of the example
├── MLP_no_cuda.cpp     # CPU-only MLP reference
├── MLP_no_cuda.hpp
├── .gitignore
└── data/               # Put your datasets here (e.g., mnist_train.csv)
```

## Building

```bash
# CUDA version
nvcc -o mlp_mnist MNIST_example.cu MLP.cu matmul.cu data_loader.cpp -O3 -arch=sm_70

# CPU reference (for comparison)
g++ -o mlp_mnist_cpu MNIST_example_cpu.cpp MLP_no_cuda.cpp data_loader.cpp -O3 -std=c++17
```

## Usage

### 1. MNIST Example (for quick testing)

```cpp
const int batch_size = 512;
MLP mlp(784, 128, 10, batch_size);   // input → hidden → output

mlp.train(train_images, train_labels, num_train_samples, epochs, lr);

float accuracy = mlp.evaluate(test_images, test_labels, num_test_samples);
```

### 2. Using for Other Tasks

You can easily adapt this MLP for **any classification or regression task**:

1. Prepare your data as:
   - `float* features` (flattened, row-major or column-major as per your loader)
   - `int* labels` (for classification) or `float* targets` (for regression)

2. Create the model with appropriate dimensions:
   ```cpp
   MLP model(input_features, hidden_size, output_size, batch_size);
   ```

3. Implement your own data loader (similar to `data_loader.cpp`) and replace the MNIST-specific parts.

## Key Components

- **Matrix Multiplication**: Shared-memory tiled matrix multiplication kernels
- **Backpropagation**: Manual implementation of gradients for weights and biases
- **Training Loop**: Shuffling + mini-batch handling inside `MLP::train()`

## Results

MNIST

Architecture:
784 → 128 → 10

Training:
Batch Size = 512
Learning Rate = 0.08
No. of Epochs = 100

Final Test Accuracy:
~97.51%

**Built as a hands-on project to master CUDA programming and neural network fundamentals.**