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

private:
        int input, hidden, output; //sizes
        int batch_size;
        float *W1, *W2; //weights
        float *b1, *b2; //biases
        float *h_pre, *h, *logits;

        //for prefetch
        int device;
        cudaMemLocation location{};

        void initializeWeights();
};

#endif