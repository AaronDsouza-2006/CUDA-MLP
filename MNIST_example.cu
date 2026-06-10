#include <iostream>

#include "data_loader.hpp"
#include "MLP.cuh"

int main()
{
    //Load datasets
    Dataset train =loadMNIST(
            "data/mnist_train.csv");

    Dataset test = loadMNIST(
            "data/mnist_test.csv");

    std::cout
        << "Train samples: "
        << train.num_samples
        << std::endl;

    std::cout
        << "Test samples: "
        << test.num_samples
        << std::endl;

    // Create Model

    const int batch_size = 512;

    MLP mlp(
        784,
        128,
        10,
        batch_size
    );

    // Train

    mlp.train(
        train.images.data(),
        train.labels.data(),
        train.num_samples,
        100,         // epochs
        0.08f       // learning rate
    );

    // Evaluate

    float test_acc =
        mlp.evaluate(
            test.images.data(),
            test.labels.data(),
            test.num_samples
        );

    std::cout
        << "Test Accuracy: "
        << test_acc
        << "%"
        << std::endl;

    // Save Model

    mlp.save_weights( "mnist_model.bin");

    std::cout
        << "Model saved to mnist_model.bin"
        << std::endl;

    return 0;
}