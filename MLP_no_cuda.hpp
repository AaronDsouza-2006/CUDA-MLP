#ifndef MLP_no_cuda_CUH
#define MLP_no_cuda_CUH

#include <vector>
#include <random>
#include <cstdlib>

class MLP_no_cuda{
    public:
        int input, hidden, output; //sizes
        //int batch_size=1;

        float *W1, *W2;
        float *b1, *b2;

        MLP_no_cuda(int input_size, int hidden_size, int output_size){
            input = input_size;
            hidden = hidden_size;
            output = output_size;

            W1 = (float*) malloc(input*hidden*sizeof(float)); 
            b1 = (float*) malloc(hidden*sizeof(float)); 
            W2 = (float*) malloc(hidden*output*sizeof(float)); 
            b2 = (float*) malloc(output*sizeof(float));
            
            initializeWeights();
        }

        ~MLP_no_cuda(){
            free(W1);
            free(b1);
            free(W2);
            free(b2);
        }

        void forward(float *x, int batch_size, float* out){
            float *h = (float*) malloc(hidden*batch_size*sizeof(float));
            
            matmul(hidden, input, input, batch_size, W1, x, h);
            bc_add(h, b1, hidden, batch_size);
            relu(h, hidden*batch_size);

            matmul(output, hidden, hidden, batch_size, W2, h, out);
            bc_add(out, b2, output, batch_size);

            free(h);
        }

    private:
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

        void relu(float* x, int n){
            for(int i=0; i<n; i++) {
                if (x[i] < 0.0f) x[i] = 0.0f;
            }
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