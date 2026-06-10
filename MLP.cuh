#ifndef MLP_CUH
#define MLP_CUH

#include <cuda_runtime.h>
#include <iostream>

class MLP{
public:
    MLP(int input_size, int hidden_size, int output_size, int input_batch_size);
    
    ~MLP();

    void forward(float* x);

    float* get_logits();

    float* softmax_batch(float* logits);

    float cross_entropy_batch(float* probs, int* labels);

    void backward(float *x, float *probs, int * labels, float lr);

    void train(float* images, int* labels, 
        int num_samples, int num_epochs, float lr);
    
    float evaluate(float* images, int* labels, int num_samples);

    void save_weights(const std::string& filename);

    void load_weights(const std::string& filename);

private:
        int input, hidden, output; //sizes
        int batch_size;
        float *W1, *W2; //weights
        float *b1, *b2; //biases
        float *h_pre, *h, *logits;
        //Gradients
        float *dW1, *dW2, *db1, *db2, *dh_pre, *dh, *dlogits; 

        //for prefetch
        cudaMemLocation location{};

        void initializeWeights();
};

#endif