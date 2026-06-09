#include <iostream>
#include <cuda_runtime.h>

#include "data_loader.hpp"
#include "MLP.cuh"

int main()
{
    Dataset train =
        loadMNIST("data/mnist_train.csv");

    std::cout
        << "Loaded "
        << train.num_samples
        << " samples\n";

    const int batch_size = 64;

    MLP model(
        784,
        128,
        10,
        batch_size
    );
    

    float* x;
    int * labels;

    cudaMallocManaged(
        &x,
        784 * batch_size * sizeof(float)
    );

    cudaMallocManaged(
        &labels,
        batch_size*sizeof(int)
    );

    // Build batch in
    // input × batch format

    for(int b = 0; b < batch_size; b++)
    {
        labels[b] = train.labels[b];
        for(int pixel = 0; pixel < 784; pixel++)
        {
            x[pixel * batch_size + b]
                = train.images[b * 784 + pixel];
        }
    }

    int device;
    cudaMemLocation location{};

    cudaGetDevice(&device);

    location.type =
        cudaMemLocationTypeDevice;

    location.id = device;

    cudaMemPrefetchAsync(
        x,
        784 * batch_size * sizeof(float),
        location,
        0,
        0
    );

    std::cout << "ready" << std::endl;

    cudaDeviceSynchronize();

    //--------------------------------------------------
    // Forward pass timing
    //--------------------------------------------------

    for(int epoch=0; epoch< 1000; epoch++){
        mlp.forward(x);
        float *probs = mlp.softmax_batch(mlp.get_logits());
        float loss = mlp.cross_entropy_batch(probs, labels);
        if(epoch % 20 == 0)
            std::cout<< "Epoch "<< epoch << 
                " Loss: "<< loss<< std::endl;
        mlp.backward(x, probs, labels);
        free(probs);
    }

    int correct = 0;

    for(int b=0; b<batch_size; b++){

        int pred = 0;

        for(int c=1; c<10; c++)
            if(probs[c*batch_size + b]
                >
            probs[pred*batch_size + b])
                pred = c;

        if(pred == labels[b])
            correct++;
    }

    std::cout<< " Loss: " << loss
    << " Acc: " << correct/batch_size
    << std::endl;

    cudaFree(x);
    cudaFree(labels);

    return 0;
}