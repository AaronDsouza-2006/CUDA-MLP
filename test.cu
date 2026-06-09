#include <iostream>
#include <cuda_runtime.h>

#include "data_loader.hpp"
#include "MLP.cuh"

int main()
{
    Dataset train =
        loadMNIST("data/mnist_train.csv");

    std::cout<<"ready"<<std::endl;

    MLP mlp(
        784,
        128,
        10,
        512
    );

    mlp.train(
        train.images.data(),
        train.labels.data(),
        train.num_samples,
        10,
        0.03f
    );
    
    mlp.save_weights("mnist_model.bin");

    Dataset test =
        loadMNIST("data/mnist_test.csv");
    
    MLP model(
        784,
        128,
        10,
        512
    );

    model.load_weights(
        "mnist_model.bin"
    );
    float accuracy = model.evaluate(
                        test.images.data(),
                        test.labels.data(),
                        test.num_samples
                    );

    std::cout
    << "Train Accuracy: "
    << accuracy
    << "%"
    << std::endl;
}