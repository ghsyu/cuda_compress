#include<stdio.h>
#include<stdlib.h>
#include<complex.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cufft.h>
#include"comfft.h"

int fft2d(float complex *src_data, float complex *dst_data, int nx, int ny){
    cufftHandle plan;    
    cufftComplex *dev_src, *dev_dst;
    //Allocate memory on the GPU and copy over the src array
    cudaSafeCall(cudaMalloc((void**) &dev_src, sizeof(cufftComplex)*nx*ny));
    cudaSafeCall(cudaMalloc((void**) &dev_dst, sizeof(cufftComplex)*nx*ny));
    cudaSafeCall(cudaMemcpy(dev_src, src_data, sizeof(cufftComplex)*nx*ny, cudaMemcpyHostToDevice));
    //Create a 2d fft plan
    //cufft functions return cufftResults, which require different error handling
    cufftSafeCall(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
    cufftSafeCall(cufftExecC2C(plan, dev_src, dev_dst, CUFFT_FORWARD));
    cudaSafeCall(cudaMemcpy(dst_data, dev_dst, sizeof(cufftComplex)*nx*ny, cudaMemcpyDeviceToHost));
    CudaCheckError();    
    //Free GPU memory
    cufftSafeCall(cufftDestroy(plan));
    cudaFree(dev_src);
    cudaFree(dev_dst);
    return 0;
}
