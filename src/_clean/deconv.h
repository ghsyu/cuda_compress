#ifndef _CLEAN_H_
#define _CLEAN_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

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
