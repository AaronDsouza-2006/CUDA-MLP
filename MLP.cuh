#ifndef MLP_CUH
#define MLP_CUH

#include <cuda_runtime.h>

class MLP{
public:
    MLP(int input_size, int hidden_size, int output_size, int input_batch_size);
    ~MLP();

    void forward(float* x);

    float* get_logits();

    float* softmax(float* logits);
    float* softmax_batch(float* logits);

    int argmax(float* values, int n);

    float cross_entropy(float* probs, int label);
    float cross_entropy_batch(float* probs, int* labels);
    void backward(float *x, float *probs, int * labels);
    void train(float* images, int* labels, 
        int num_samples, int num_epochs, float lr);

private:
        int input, hidden, output; //sizes
        int batch_size;
        float *W1, *W2; //weights
        float *b1, *b2; //biases
        float *h_pre, *h, *logits;
        //Gradients
        float *dW1, *dW2, *db1, *db2, *dh_pre, *dh, *dlogits; 
        
        //float lr = 0.1;

        //for prefetch
        int device;
        cudaMemLocation location{};

        void initializeWeights();
};

#endif