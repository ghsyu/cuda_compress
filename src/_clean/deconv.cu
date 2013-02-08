/*
 * Some additional deconvolution functions for AIPY, written in C++ and CUDA.  These are
 * mostly for speed-critical applications. 
 *
 * Author: Aaron Parsons, Gilbert Hsyu
 */

#include "deconv.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


__global__ void clean2dc(int *dim1_p, int *dim2_p, int *argmax1_p, int *argmax2_p, float *stepr_p, \
                        float *stepi_p, float *ker, float *res, float *val_arr){ 
    int dim1 = *dim1_p;
    int dim2 = *dim2_p;
    int argmax1 = *argmax1_p;
    int argmax2 = *argmax2_p;
    float stepr = *stepr_p;
    float stepi = *stepi_p;
    float valr, vali;
    int gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    int n2 = threadIdx.y + blockIdx.y * blockDim.y;
    if ((n1 < dim1) && (n2 < dim2)){
        int wrap_n1 = (n1 + argmax1) % dim1;
        int wrap_n2 = (n2 + argmax2) % dim2;
        res[2*(wrap_n1 + wrap_n2*dim1)]     -= (ker[2*(n1 + n2*dim1)] * stepr - ker[2*(n1 + n2*dim1)+1] * stepi);
        res[2*(wrap_n1 + wrap_n2*dim1) + 1] -= (ker[2*(n1 + n2*dim1)] * stepi + ker[2*(n1 + n2*dim1)+1] * stepr);
        valr = res[2*(wrap_n1 + wrap_n2*dim1)];
        vali = res[2*(wrap_n1 + wrap_n2*dim1) + 1];
        val_arr[2*(n1 + blockDim.x*gridx*n2)]     = valr;
        val_arr[2*(n1 + blockDim.x*gridx*n2) + 1] = vali;
    }
    return;
}

//   ____ _                  ____     _      
//  / ___| | ___  __ _ _ __ |___ \ __| | ___ 
// | |   | |/ _ \/ _` | '_ \  __) / _` |/ __|
// | |___| |  __/ (_| | | | |/ __/ (_| | (__ 
//  \____|_|\___|\__,_|_| |_|_____\__,_|\___|  
// Does a 2d complex-valued clean
float *clean_2d_c_GPU(float *res, float *ker,
        double gain, int maxiter, \
        int argmax1, int argmax2, \
        int stop_if_div, \
        float *stepr_p, float *stepi_p, \
        int ker_len, int res_len, int dim1, int dim2, float* retval) {
    int gridx, gridy;
    float *dev_ker, *dev_res, *dev_stepr, *dev_stepi, *dev_val_arr;
    int *dev_argmax1, *dev_argmax2,
        *dev_dim1, *dev_dim2;
    float stepr = *stepr_p;
    float stepi = *stepi_p;
    //Ceiling division of dim1/BLOCKSIZEX and dim2/BLOCKSIZEY
    gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
    gridy = (dim2 % BLOCKSIZEY == 0) ? dim2/BLOCKSIZEY : dim2/BLOCKSIZEY + 1;
    dim3 grid(gridx, gridy);
    dim3 blocksize(BLOCKSIZEX, BLOCKSIZEY);
    //arr_size will be different from res_len if a dimension of res is not a multiple of 16.
    int arr_size = BLOCKSIZEX*BLOCKSIZEY*gridx*gridy;
    float *val_arr;
    val_arr = (float *)malloc(2*arr_size*sizeof(float));
    if (val_arr == NULL){
        exit (EXIT_FAILURE);
    }
    
    CudaSafeCall(cudaMalloc((void**) &dev_ker,      ker_len));
    CudaSafeCall(cudaMalloc((void**) &dev_res,      res_len));
    CudaSafeCall(cudaMalloc((void**) &dev_dim1,     sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_dim2,     sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_argmax1,  sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_argmax2,  sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_stepr,    sizeof(float)));
    CudaSafeCall(cudaMalloc((void**) &dev_stepi,    sizeof(float)));
    CudaSafeCall(cudaMalloc((void**) &dev_val_arr,  2*sizeof(float)*(gridx*BLOCKSIZEX*gridy*BLOCKSIZEY)));
    
    CudaSafeCall(cudaMemcpy(dev_ker,      ker,      ker_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_res,      res,      res_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_dim1,     &dim1,    sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_dim2,     &dim2,    sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_argmax1,  &argmax1, sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_argmax2,  &argmax2, sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_stepr,    &stepr,   sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_stepi,    &stepi,   sizeof(float), cudaMemcpyHostToDevice));

    // Take next step and compute score
    clean2dc<<<grid, blocksize>>>(dev_dim1, dev_dim2, dev_argmax1, dev_argmax2, dev_stepr, \
                                dev_stepi, dev_ker, dev_res, dev_val_arr);
    cudaThreadSynchronize();
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(val_arr, dev_val_arr, 2*sizeof(float)*arr_size,  cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(res,     dev_res,     res_len,                   cudaMemcpyDeviceToHost));
    cudaFree(dev_ker);
    cudaFree(dev_res);
    cudaFree(dev_dim1);
    cudaFree(dev_dim2);
    cudaFree(dev_argmax1);
    cudaFree(dev_argmax2);
    cudaFree(dev_stepr);
    cudaFree(dev_stepi);
    cudaFree(dev_val_arr);
    for(int i = 0, j = 0; i < 2*arr_size; i++){
        if(i % (2*BLOCKSIZEX) < 2*dim1 && i/(2*BLOCKSIZEX) < 2*dim2){
            retval[j] = val_arr[i];
            j++;
        }
    }
    free(val_arr);
    return retval;
}