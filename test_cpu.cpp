#include <iostream>
#include "data_loader.hpp"
#include "MLP_no_cuda.hpp"

int main(){

    Dataset train =
        loadMNIST("data/mnist_train.csv");

    const int batch_size = 32;

    float* x =
        (float*)malloc(
            784 * batch_size *
            sizeof(float)
        );

    int* labels =
        (int*)malloc(
            batch_size *
            sizeof(int)
        );
    std::cout<<"ready";
    // build ONE fixed batch
    for(int b=0; b<batch_size; b++){

        labels[b] = train.labels[b];

        for(int pixel=0; pixel<784; pixel++){

            x[pixel*batch_size + b]
                =
                train.images[
                    b*784 + pixel
                ];
        }
    }

    MLP_no_cuda mlp(
        784,
        128,
        10,
        batch_size
    );
    
    std::cout<<"MLP and batch created";
    for(int epoch=0; epoch<1000; epoch++){

        mlp.forward(x);

        float* probs =
            mlp.softmax_batch(
                mlp.get_logits()
            );

        float loss =
            mlp.cross_entropy_batch(
                probs,
                labels
            );

        if(epoch % 10 == 0)
            std::cout
                << "Epoch "
                << epoch
                << " Loss: "
                << loss
                << std::endl;

        mlp.backward(
            x,
            probs,
            labels
        );

        free(probs);
    }

    free(x);
    free(labels);

    return 0;
}