#ifndef _CLEAN_H_
#define _CLEAN_H_


#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

float *clean_2d_c_GPU(float *, float *, double, int, int, int, \
					  int, float *, float*, int, int, int, int, float *);

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
