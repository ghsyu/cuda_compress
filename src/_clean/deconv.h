#ifndef _CLEAN_H_
#define _CLEAN_H_


#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "sharedmem.cuh"



//GPU code using templates must be defined in the header file
template<typename T> 
    __global__ void clean2drGPU (int *dim1_p, int *dim2_p, int *argmax1_p, int *argmax2_p, T *step_p, T *ker, T *res,
                               int *pos_def, T *nscore, T *val_arr){ 
        int dim1 = *dim1_p;
        int dim2 = *dim2_p;
        int argmax1 = *argmax1_p;
        int argmax2 = *argmax2_p;
        float step = *step_p;
        int gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
        T val, mval;
        //Array for accumulating the nscores of the block
		SharedMemory<T> shared;
		T* s_data = shared.getPointer();
        int tid = threadIdx.x + BLOCKSIZEX*threadIdx.y;
        int n1 = threadIdx.x + blockIdx.x * blockDim.x;
        int n2 = threadIdx.y + blockIdx.y * blockDim.y;
        int wrap_n1 = (n1 + argmax1) % dim1;
        int wrap_n2 = (n2 + argmax2) % dim2;
        res[wrap_n1 + wrap_n2*sizeof(T)*dim1] -= ker[n1 + n2*sizeof(T)*dim1]*step;
        val = res[wrap_n1 + wrap_n2*sizeof(T)*dim1];
        mval = val * val;
        s_data[tid] = mval;
        for (int s = blockDim.x * blockDim.y/2; s>0; s>>=1){
            if (tid < s){
                s_data[tid] += s_data[tid + s];
            }
        }
        __syncthreads();
        if (tid == 0){
            nscore[blockIdx.x + blockIdx.y*gridx] = s_data[0];
        __syncthreads();
        val_arr[n1 + blockDim.x * n2] = val;
        }
		return;
    }

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

static void __cudaCheckError( const char *file, const int line )
{

	cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( EXIT_FAILURE );
    }
}

#define CudaSafeCall( err ) (HandleError( err, __FILE__, __LINE__ ))
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


#endif /* _CLEAN_H_ */
