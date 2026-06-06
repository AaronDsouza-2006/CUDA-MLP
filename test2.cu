#include <iostream>
#include "data_loader.hpp"
#include "MLP_no_cuda.hpp"

int main()
{
    Dataset train =
        loadMNIST("MNIST_data/mnist_train.csv");

    std::cout
        << "Loaded "
        << train.num_samples
        << " samples\n";

    MLP_no_cuda model(
        784,
        128,
        10
    );

    const int batch_size = 64;

    float x[784 * batch_size];
    float output[10 * batch_size];

    for(int b = 0; b < batch_size; b++)
    {
        for(int pixel = 0; pixel < 784; pixel++)
        {
            x[pixel * batch_size + b]
                = train.images[b * 784 + pixel];
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    model.forward(
        x,
        batch_size,
        output
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout
        << "Forward time: "
        << ms
        << " ms\n";

    for(int b = 0; b < batch_size; b++)
    {
        std::cout
            << "\nImage "
            << b
            << ":\n";

        for(int c = 0; c < 10; c++)
        {
            std::cout
                << output[c * batch_size + b]
                << "\n";
        }
    }

    return 0;
}