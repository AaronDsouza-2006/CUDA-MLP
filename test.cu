#include <iostream>
#include <cuda_runtime.h>
#include "data_loader.hpp"
#include "MLP.cuh"

int main()
{
    Dataset train =
        loadMNIST("MNIST_data/mnist_train.csv");

    std::cout
        << "Loaded "
        << train.num_samples
        << " samples\n";

    MLP model(
        784,
        128,
        10
    );

    float* x;
    float* out;
    int batch_size=64;

    cudaMallocManaged(
        &x,
        784*batch_size*sizeof(float)
    );

    cudaMallocManaged(
        &out,
        10*batch_size*sizeof(float)
    );

    int device;
    cudaMemLocation location{};
    cudaGetDevice(&device);
    location.type = cudaMemLocationTypeDevice;
    location.id = device;

    for(int b=0; b<batch_size; b++)
    {
        for(int pixel=0; pixel<784; pixel++)
        {
            x[pixel*batch_size + b]
                = train.images[b*784 + pixel];
        }
    }
    
    cudaMemPrefetchAsync(x, 784*batch_size*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(out, 10*batch_size*sizeof(float), location, 0, 0);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    model.forward(
        x,
        batch_size,
        out
    );

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel time: "<< ms << " ms\n";

    std::cout
        << "\nNetwork output:\n";

    for(int b=0; b<batch_size; b++)
    {
        std::cout << "\nImage " << b << ":\n";

        for(int c=0; c<10; c++)
        {
            std::cout
                << out[c*batch_size + b]
                << "\n";
        }
    }

    cudaFree(x);
    cudaFree(out);

    return 0;
}