#ifndef MLP_no_cuda_CUH
#define MLP_no_cuda_CUH

#include <cuda_runtime.h>

class MLP_no_cuda{
public:
    MLP_no_cuda(int input_size, int hidden_size, int output_size, int input_batch_size);
    ~MLP_no_cuda();

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

        void initializeWeights();
        void bc_add(float* mat, float* vec, int rows, int cols);
        void relu(float* x, int n, float *y);
        void matmul(int A, int B, int M, int N, float *x, float *y, float *z);
};

#endif