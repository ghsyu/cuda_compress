/*
 * Some additional deconvolution functions for AIPY, written in C++ and CUDA.  These are
 * mostly for speed-critical applications. 
 *
 * Author: Aaron Parsons, Gilbert Hsyu
 */

#include "deconv.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
//XXX The area matrix is currently not working
__global__ void sum_max(unsigned int n, float *g_nscore_i, float *g_max_i, int* g_max_idx_i, \
                        float *g_nscore_o, float *g_max_o, int *g_max_idx_o){
    extern __shared__ float smem[];
    float *s_nscore = smem;
    float *s_max    = &s_nscore[blockDim.x];
    int *s_max_idx  = (int *) &s_max[2*blockDim.x];
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + tid;
    if (i < n){
        s_nscore[tid]      = g_nscore_i[i];
        s_max[2*tid]       = g_max_i[2*i];
        s_max[2*tid+1]     = g_max_i[2*i+1];
        s_max_idx[2*tid]   = g_max_idx_i[2*i];
        s_max_idx[2*tid+1] = g_max_idx_i[2*i+1];
    } else {
        s_nscore[tid]      = 0;
        s_max[2*tid]       = 0;
        s_max[2*tid+1]     = 0;
        s_max_idx[2*tid]   = 0;
        s_max_idx[2*tid+1] = 0;
    }
    __syncthreads();
    /* if Re(i)^2 + Im(i)^2 > Re(j)^2 + Im(j)**2, copy j to i (where i and j are elements of s_max)
       and also copy the associated s_max_idx element
       The loop will run until max(Re(i)^2 + Im(i)^2) is the 0th element of s_max */
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            s_nscore[tid] += s_nscore[tid + s];
            if ((s_max[2*tid]*s_max[2*tid] + s_max[2*tid+1]*s_max[2*tid+1]) < \
                (s_max[2*(tid+s)]*s_max[2*(tid+s)] + s_max[2*(tid+s)+1]*s_max[2*(tid+s)+1])){
                s_max[2*tid]         = s_max[2*(tid+s)];
                s_max[2*tid+1]       = s_max[2*(tid+s)+1];
                s_max_idx[2*tid]     = s_max_idx[2*(tid+s)];
                s_max_idx[2*tid+1]   = s_max_idx[2*(tid+s)+1];
            }
        }
        __syncthreads();
    }
    if (tid == 0){
        g_nscore_o[blockIdx.x]        = s_nscore[0];
        g_max_o[2*blockIdx.x]         = s_max[0];
        g_max_o[2*blockIdx.x+1]       = s_max[1];
        g_max_idx_o[2*blockIdx.x]     = s_max_idx[0];
        g_max_idx_o[2*blockIdx.x+1]   = s_max_idx[1];
    }
}
__global__ void clean2dc(unsigned int dim1, unsigned int dim2, unsigned int argmax1, unsigned int argmax2, float stepr, \
                        float stepi, float *ker, float *res, int * area, float *g_nscore, \
                        float *g_max, int *g_max_idx){
    float valr, vali;
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    int n2 = threadIdx.y + blockIdx.y * blockDim.y;
    int i = n1 * dim2 + n2;
    if ((n1 < dim1) && (n2 < dim2)){
        int wrap_n1 = (n1 + argmax1) % dim1;
        int wrap_n2 = (n2 + argmax2) % dim2;
        if (area[wrap_n1*dim2 + wrap_n2] == 1){
            res[2*(wrap_n1*dim2 + wrap_n2)]     -= (ker[2*(i)] * stepr - ker[2*(i)+1] * stepi);
            res[2*(wrap_n1*dim2 + wrap_n2) + 1] -= (ker[2*(i)] * stepi + ker[2*(i)+1] * stepr); 
            valr = res[2*(wrap_n1*dim2 + wrap_n2)];
            vali = res[2*(wrap_n1*dim2 + wrap_n2) + 1];
            g_nscore[i] = valr*valr+vali*vali;
            g_max[2*i] = valr;
            g_max[2*i+1] = vali;
            g_max_idx[2*i] = wrap_n1;
            g_max_idx[2*i+1] = wrap_n2;
        } else {
          g_nscore[i]      = 0;
          g_max[2*i]       = 0;
          g_max[2*i+1]     = 0;
          g_max_idx[2*i]   = 0;
          g_max_idx[2*i+1] = 0;
        }
    }
    return;
}
//Helper functions

int gpu_set_up(float **dev_ker,    float **dev_res, int **dev_area, \
               float **g_nscore_i, float **g_max_i, int **g_max_idx_i, \
               float **g_nscore_o, float **g_max_o, int **g_max_idx_o, \
               float *ker, float *res, int *area, int dim1, int dim2, int ker_len, int res_len, int area_len){
    CudaSafeCall(cudaMalloc((void**) dev_ker,      ker_len));
    CudaSafeCall(cudaMalloc((void**) dev_res,      res_len));
    CudaSafeCall(cudaMalloc((void**) dev_area,     area_len));
    CudaSafeCall(cudaMalloc((void**) g_nscore_i,   sizeof(float)*dim1*dim2));
    CudaSafeCall(cudaMalloc((void**) g_max_i,      2*sizeof(float)*dim1*dim2));
    CudaSafeCall(cudaMalloc((void**) g_max_idx_i,  2*sizeof(int)*dim1*dim2));
    CudaSafeCall(cudaMalloc((void**) g_nscore_o,   sizeof(float)*dim1*dim2));
    CudaSafeCall(cudaMalloc((void**) g_max_o,      2*sizeof(float)*dim1*dim2));
    CudaSafeCall(cudaMalloc((void**) g_max_idx_o,  2*sizeof(int)*dim1*dim2));
    CudaSafeCall(cudaMemcpy(*dev_ker,      ker,      ker_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(*dev_res,      res,      res_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(*dev_area,     area,     area_len,      cudaMemcpyHostToDevice));
    return 0;
}

int gpu_free(float *dev_ker, float *dev_res, int *dev_area, \
             float *g_nscore_i, float *g_max_i, int *g_max_idx_i, \
             float *g_nscore_o, float *g_max_o, int *g_max_idx_o){
    cudaFree(dev_ker);
    cudaFree(dev_res);
    cudaFree(g_nscore_i);
    cudaFree(g_max_i);
    cudaFree(g_max_idx_i);
    cudaFree(g_nscore_o);
    cudaFree(g_max_o);
    cudaFree(g_max_idx_o);
    return 0;
}


//   ____ _                  ____     _      
//  / ___| | ___  __ _ _ __ |___ \ __| | ___ 
// | |   | |/ _ \/ _` | '_ \  __) / _` |/ __|
// | |___| |  __/ (_| | | | |/ __/ (_| | (__ 
//  \____|_|\___|\__,_|_| |_|_____\__,_|\___|  
// Does a 2d complex-valued clean
float *clean_2d_c_GPU(float *res, float *ker, int * area, \
        double gain, int maxiter, \
        int stop_if_div, \
        float stepr, float stepi, int argmax1, int argmax2, \
        int ker_len, int res_len, int area_len, int dim1, int dim2,
        float *nscore_p, float *maxr_p, float *maxi_p, int *nargmax1_p, int *nargmax2_p, \
        float *dev_ker, float *dev_res, int *dev_area, \
        float *g_nscore_i, float *g_max_i, int *g_max_idx_i, \
        float *g_nscore_o, float *g_max_o, int *g_max_idx_o) {
    int gridx, gridy;
    float max_p[2];
    int max_idx_p[2];
    int gridsize;
    //Ceiling division of dim1/BLOCKSIZEX and dim2/BLOCKSIZEY
    gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
    gridy = (dim2 % BLOCKSIZEY == 0) ? dim2/BLOCKSIZEY : dim2/BLOCKSIZEY + 1;
    dim3 grid(gridx, gridy);
    dim3 blocksize(BLOCKSIZEX, BLOCKSIZEY);
    CudaCheckError();

    // Take next step and compute score
    clean2dc<<<grid, blocksize>>>(dim1, dim2, argmax1, argmax2, stepr, \
                                stepi, dev_ker, dev_res, dev_area, g_nscore_i, g_max_i, g_max_idx_i);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(res, dev_res, res_len, cudaMemcpyDeviceToHost));
    //Make the kernel invocation 1D
    int bsize = BLOCKSIZEX*BLOCKSIZEY;
    int smemsize = 3*BLOCKSIZEX*BLOCKSIZEY*sizeof(float)+2*BLOCKSIZEX*BLOCKSIZEY*sizeof(int);
    int len = dim1*dim2; //The total number of elements left to reduce
    while(len > bsize){
        gridsize = (len % bsize == 0) ? len/bsize : len/bsize + 1;
        sum_max<<<gridsize, bsize, smemsize>>>(len, g_nscore_i, g_max_i, g_max_idx_i, g_nscore_o, g_max_o, g_max_idx_o);
        CudaCheckError();
        len = gridsize;
        CudaSafeCall(cudaMemcpy(g_nscore_i, g_nscore_o,   sizeof(float)*len,   cudaMemcpyDeviceToDevice));
        CudaSafeCall(cudaMemcpy(g_max_i, g_max_o,         2*sizeof(float)*len, cudaMemcpyDeviceToDevice));
        CudaSafeCall(cudaMemcpy(g_max_idx_i, g_max_idx_o, 2*sizeof(int)*len,   cudaMemcpyDeviceToDevice));
    }

    gridsize = (len % bsize == 0) ? len/bsize : len/bsize + 1;
    sum_max<<<gridsize, bsize, smemsize>>>(len, g_nscore_i, g_max_i, g_max_idx_i, g_nscore_o, g_max_o, g_max_idx_o);
    CudaCheckError();
    len = gridsize;
    CudaSafeCall(cudaMemcpy(g_nscore_i, g_nscore_o,   sizeof(float)*dim1*dim2,   cudaMemcpyDeviceToDevice));
    CudaSafeCall(cudaMemcpy(g_max_i, g_max_o,         2*sizeof(float)*dim1*dim2, cudaMemcpyDeviceToDevice));
    CudaSafeCall(cudaMemcpy(g_max_idx_i, g_max_idx_o, 2*sizeof(int)*dim1*dim2,   cudaMemcpyDeviceToDevice));
    CudaSafeCall(cudaMemcpy(nscore_p, g_nscore_o, sizeof(float), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(max_p, g_max_o, 2*sizeof(float), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(max_idx_p, g_max_idx_o, 2*sizeof(int), cudaMemcpyDeviceToHost));
    *nargmax1_p = max_idx_p[0];
    *nargmax2_p = max_idx_p[1];
    *maxr_p = max_p[0];
    *maxi_p = max_p[1];
    return 0;
}