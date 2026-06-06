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

    cudaMallocManaged(
        &x,
        784 * batch_size * sizeof(float)
    );

    // Build batch in
    // input × batch format

    for(int b = 0; b < batch_size; b++)
    {
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

    cudaDeviceSynchronize();

    //--------------------------------------------------
    // Forward pass timing
    //--------------------------------------------------

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    model.forward(x);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;

    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );

    std::cout
        << "\nForward time: "
        << ms
        << " ms\n";

    //--------------------------------------------------
    // Softmax
    //--------------------------------------------------

    float* probs =
        model.softmax_batch(
            model.get_logits()
        );

    //--------------------------------------------------
    // Check softmax sums
    //--------------------------------------------------

    std::cout
        << "\nSoftmax column sums:\n";

    for(int b = 0; b < 5; b++)
    {
        float sum = 0;

        for(int c = 0; c < 10; c++)
        {
            sum +=
                probs[
                    c * batch_size + b
                ];
        }

        std::cout
            << "sample "
            << b
            << " -> "
            << sum
            << "\n";
    }

    //--------------------------------------------------
    // Loss
    //--------------------------------------------------

    float loss =
        model.cross_entropy_batch(
            probs,
            train.labels.data()
        );

    std::cout
        << "\nBatch loss = "
        << loss
        << "\n";

    //--------------------------------------------------
    // Predictions
    //--------------------------------------------------

    int correct = 0;

    std::cout
        << "\nFirst 10 predictions:\n";

    for(int b = 0; b < 10; b++)
    {
        int pred = 0;

        for(int c = 1; c < 10; c++)
        {
            if(
                probs[
                    c * batch_size + b
                ]
                >
                probs[
                    pred * batch_size + b
                ]
            )
            {
                pred = c;
            }
        }

        std::cout
            << "label="
            << train.labels[b]
            << " pred="
            << pred
            << "\n";

        if(pred == train.labels[b])
            correct++;
    }

    std::cout
        << "\nFirst-10 accuracy = "
        << correct
        << "/10\n";

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------

    free(probs);

    cudaFree(x);

    return 0;
}