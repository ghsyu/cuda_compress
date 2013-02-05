
/*
 * Some additional deconvolution functions for AIPY, written in C++ and CUDA.  These are
 * mostly for speed-critical applications. 
 *
 * Author: Aaron Parsons, Gilbert Hsyu
 */

#include "deconv.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


__global__ void clean2dr(int *dim1_p, int *dim2_p, int *argmax1_p, int *argmax2_p, float *stepr_p, \
                         float *stepi_p, float *ker, float *res, int *pos_def, float *nscore, \
                         float *val_arr){ 
    int dim1 = *dim1_p;
    int dim2 = *dim2_p;
    int argmax1 = *argmax1_p;
    int argmax2 = *argmax2_p;
    float stepr = *stepr_p;
    float stepi = *stepi_p;
    int gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
    float valr, vali;
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    int n2 = threadIdx.y + blockIdx.y * blockDim.y;
    int wrap_n1 = (n1 + argmax1) % dim1;
    int wrap_n2 = (n2 + argmax2) % dim2;
    res[2*(wrap_n1 + wrap_n2*sizeof(float)*dim1)] -= ker[n1 + n2*sizeof(float)*dim1]*stepr;
    res[2*(wrap_n1 + wrap_n2*sizeof(float)*dim1) + 1] -= ker[n1 + n2*sizeof(float)*dim1]*stepi;
    valr = res[wrap_n1 + wrap_n2*sizeof(float)*dim1];
    vali = res[wrap_n1 + wrap_n2*sizeof(float)*dim1 + 1];
    val_arr[2*(n1 + blockDim.x*gridx*n2)]     = valr;
    val_arr[2*(n1 + blockDim.x*gridx*n2) + 1] = vali;
    return;
}

//   ____ _                  ____     _      
//  / ___| | ___  __ _ _ __ |___ \ __| | ___ 
// | |   | |/ _ \/ _` | '_ \  __) / _` |/ __|
// | |___| |  __/ (_| | | | |/ __/ (_| | (__ 
//  \____|_|\___|\__,_|_| |_|_____\__,_|\___|  
// Does a 2d complex-valued clean
float *clean_2d_c_GPU(float *res, float *ker,
        float *mdl, float *area, \
        double gain, int maxiter, \
        float argmax1, float argmax2, \
        int stop_if_div, int pos_def, \
        float *stepr_p, float *stepi_p, \
        int ker_len, int res_len, int dim1, int dim2, float* val_arr) {
    int gridx, gridy;
    float *dev_ker, *dev_res, *dev_stepr, *dev_stepi, *dev_nscore, *dev_val_arr;
    int *dev_argmax1, *dev_argmax2,
        *dev_dim1, *dev_dim2, *dev_pos_def;
    float stepr = *stepr_p;
    float stepi = *stepi_p;
    
    CudaSafeCall(cudaMalloc((void**) &dev_ker,      ker_len));
    CudaSafeCall(cudaMalloc((void**) &dev_res,      res_len));
    CudaSafeCall(cudaMalloc((void**) &dev_dim1,     sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_dim2,     sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_argmax1,  sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_argmax2,  sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_stepr,    sizeof(float)));
    CudaSafeCall(cudaMalloc((void**) &dev_stepi,    sizeof(float)));
    CudaSafeCall(cudaMalloc((void**) &dev_pos_def,  sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_nscore,   sizeof(float)*(dim1*dim2/(BLOCKSIZEX * BLOCKSIZEY)+1)));
    CudaSafeCall(cudaMalloc((void**) &dev_val_arr,  2*sizeof(float)*(dim1*dim2)));
    
    CudaSafeCall(cudaMemcpy(dev_ker,      ker,      ker_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_res,      res,      res_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_dim1,     &dim1,    sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_dim2,     &dim2,    sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_argmax1,  &argmax1, sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_argmax2,  &argmax2, sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_stepr,    &stepr,   sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_stepi,    &stepi,   sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_pos_def,  &pos_def, sizeof(int),   cudaMemcpyHostToDevice));
    //Ceiling division of dim1/BLOCKSIZEX and dim2/BLOCKSIZEY
    gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
    gridy = (dim2 % BLOCKSIZEY == 0) ? dim2/BLOCKSIZEY : dim2/BLOCKSIZEY + 1;
    dim3 grid(gridx, gridy);
    dim3 blocksize(BLOCKSIZEX, BLOCKSIZEY);
    // Take next step and compute score
    clean2dr<<<grid, blocksize>>>(dev_dim1, dev_dim2, dev_argmax1, dev_argmax2, dev_stepr, \
                                dev_stepi, dev_ker, dev_res, dev_pos_def, dev_nscore, dev_val_arr);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(val_arr, dev_val_arr, 2*sizeof(float)*dim1*dim2, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(res,     dev_res,     res_len,                   cudaMemcpyDeviceToHost));
    cudaFree(dev_ker);
    cudaFree(dev_res);
    cudaFree(dev_dim1);
    cudaFree(dev_dim2);
    cudaFree(dev_argmax1);
    cudaFree(dev_argmax2);
    cudaFree(dev_stepr);
    cudaFree(dev_stepi);
    cudaFree(dev_nscore);
    cudaFree(dev_pos_def);
    cudaFree(dev_val_arr);
    return val_arr;
}