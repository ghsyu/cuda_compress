#ifndef _FFT_H_
#define _FFT_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

int fft2d(cufftComplex *, cufftComplex *, int, int);

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

static void HandleCufftError(cufftResult  err,
                         const char *file,
                         int line ) {
    if (err != 0) {
        printf( "Cufft error code %d in %s at line %d\n", err ,
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
        exit( -1 );
    }
}

#define cudaSafeCall( err ) (HandleError( err, __FILE__, __LINE__ ))
#define cufftSafeCall( err ) (HandleCufftError( err, __FILE__, __LINE__))
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


#ifdef __cplusplus
}
#endif

#endif /* _TEST_CUDA_H_ */
