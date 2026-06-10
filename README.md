# CUDA-MLP

A **from-scratch** implementation of a Multi-Layer Perceptron (MLP) neural network using **CUDA/C++** for high-performance GPU training. 

This project demonstrates efficient custom CUDA kernels for matrix operations, forward & backward propagation, and is implemented using MNIST classification as an example.

## Features

- CUDA-accelerated forward and backward passes
- 2-layers with one hidden layer.
- Tiled matrix multiplication with shared memory for performance
- ReLU activation + Softmax + Cross-Entropy loss
- Mini-batch Gradient Descent
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
├── MNIST_example.cu    # MNIST Example classification
├── MNIST_example_cpu.cpp # CPU version of the example
├── MLP_no_cuda.cpp     # CPU-only MLP reference
├── MLP_no_cuda.hpp
```

## Building

```bash
# CUDA version
nvcc MNIST_example.cu MLP.cu matmul.cu data_loader.cpp -o mlp_mnist

# CPU reference (for comparison)
g++ MNIST_example_cpu.cpp MLP_no_cuda.cpp data_loader.cpp -o mlp_mnist_cpu
```

## Usage

### 1. MNIST Example (for quick testing)

```cpp
MLP mlp(input_size, hidden_size, output_size, batch_size);   

mlp.train(train_images, train_labels, num_train_samples, epochs, lr);

accuracy = mlp.evaluate(test_images, test_labels, num_test_samples);
```

### 2. Using for Other Tasks

You can easily adapt this MLP for **any classification or regression task**:

1. Prepare your data as:
   - `float* features` (flattened)
   - `int* labels` 

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
