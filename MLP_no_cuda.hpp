#ifndef MLP_no_cuda_CUH
#define MLP_no_cuda_CUH

#include <vector>
#include <random>
#include <cstdlib>

class MLP_no_cuda{
    public:
        MLP_no_cuda(int input_size, int hidden_size, int output_size, int input_batch_size){
            input = input_size;
            hidden = hidden_size;
            output = output_size;
            batch_size = input_batch_size;

            W1 = (float*) malloc(input*hidden*sizeof(float)); 
            b1 = (float*) malloc(hidden*sizeof(float)); 
            W2 = (float*) malloc(hidden*output*sizeof(float)); 
            b2 = (float*) malloc(output*sizeof(float));
        
            h_pre = (float*) malloc(hidden*batch_size*sizeof(float));
            h = (float*) malloc(hidden*batch_size*sizeof(float));
            logits = (float*) malloc(output*batch_size*sizeof(float));
            initializeWeights();
        }

        ~MLP_no_cuda(){
            free(W1);
            free(b1);
            free(W2);
            free(b2);

            free(h_pre);
            free(h);
            free(logits);
        }

        void forward(float *x){
            matmul(hidden, input, input, batch_size, W1, x, h_pre);
            bc_add(h_pre, b1, hidden, batch_size);
            relu(h_pre, hidden*batch_size, h);

            matmul(output, hidden, hidden, batch_size, W2, h, logits);
            bc_add(logits, b2, output, batch_size);
        }
        
        float* softmax(float* logits, int output_size){
            int max_val_idx = argmax(logits, output_size);
            float sum = 0, *probs;
            probs = (float*) malloc(output_size*sizeof(float));

            for(int i=0; i<output_size; i++){
                probs[i] = std::exp(logits[i]-logits[max_val_idx]);
                sum += probs[i];
            }

            for(int i=0; i<output_size; i++) probs[i]/=sum;

            return probs;
        }

        float* softmax_batch(float* logits){
            float* probs =
                (float*) malloc(output*batch_size*sizeof(float));

            for(int b=0; b<batch_size; b++){
                float max_val = logits[b];

                for(int c=1; c<output; c++){
                    float val = logits[c*batch_size + b];
                    if(val > max_val) max_val = val;
                }

                float sum = 0;

                for(int c=0; c<output; c++){
                    probs[c*batch_size + b] =
                        std::exp(logits[c*batch_size + b] - max_val);

                    sum += probs[c*batch_size + b];
                }

                for(int c=0; c<output; c++)
                    probs[c*batch_size + b] /= sum;
            }

            return probs;
        }

        int argmax(float* values, int n){
            int max_val_idx = 0;
            for(int i=1; i<n; i++)
                if (values[i] > values[max_val_idx]) max_val_idx = i;
            
            return max_val_idx;
        }

        float cross_entropy(float* probs, int label){
            return -std::log(probs[label] + 1e-8f);
        }

        float cross_entropy_batch(float* probs, int* labels){
            float loss = 0;

            for(int b=0; b<batch_size; b++)
            {
                int label = labels[b];
                loss += -std::log(probs[label * batch_size + b] + 1e-8f);
            }

            return loss / batch_size;
        }

        float* get_logits(){
            return logits;
        }

    private:
        int input, hidden, output; //sizes
        int batch_size;

        float *W1, *W2;
        float *b1, *b2;

        float *h_pre, *h, *logits;

        void initializeWeights(){
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0f, 0.01f);

            for(int i=0; i<input*hidden; i++) W1[i] = dist(gen);
            for(int i=0; i<hidden*output; i++) W2[i] = dist(gen);

            for(int i=0; i<hidden; i++) b1[i] = 0.0f;
            for(int i=0; i<output; i++) b2[i] = 0.0f;
        }

        //broadcast addition of matrix with a vector
        void bc_add(float* mat, float* vec, int rows, int cols){
            for(int row=0; row<rows; row++){
                for(int col=0; col<cols; col++){
                    mat[row*cols + col] += vec[row];
                }
            }
        }

        void relu(float* x, int n, float *y){
            for(int i=0; i<n; i++)
                y[i] = (x[i] < 0.0f) ? 0.0f : x[i];
        }

        void matmul(int A, int B, int M, int N, float *x, float *y, float *z){
            if(B != M)
                return;

            for(int row = 0; row < A; row++)
            {
                for(int col = 0; col < N; col++)
                {
                    float sum = 0.0f;
                    
                    for(int k = 0; k < B; k++)
                        sum += x[row * B + k] * y[k * N + col];

                    z[row * N + col] = sum;
                }
            }
        }
};

#endif