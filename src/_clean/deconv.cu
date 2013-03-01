/*
 * Some additional deconvolution functions for AIPY, written in C++ and CUDA.  These are
 * mostly for speed-critical applications. 
 *
 * Author: Aaron Parsons, Gilbert Hsyu
 */

#include "deconv.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void sum_max(float *g_nscore, float *g_max, int* g_max_idx){
    extern __shared__ float s_nscore[];
    float *s_max = s_nscore + blockDim.x;
    int *s_max_idx =(int *) s_nscore + 3*blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + tid;
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            s_nscore[tid] += s_nscore[tid + s];
            if ((s_max[2*tid]*s_max[2*tid] + s_max[2*tid+1]*s_max[2*tid+1]) < \
                (s_max[2*(tid+s)]*s_max[2*(tid+s)] + s_max[2*(tid+s)+1]*s_max[2*(tid+s)+1])){
                s_max[2*tid]         = s_max[2*(tid+s)];
                s_max[2*tid+1]       = s_max[2*(tid+s)+1];
                s_max_idx[2*tid]     = s_max_idx[2*(tid+s)];
                s_max_idx[2*tid + 1] = s_max_idx[2*(tid+s)+1];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0){
        g_nscore[i]        = s_nscore[0];
        g_max[2*i]         = s_max[0];
        g_max[2*i+1]       = s_max[1];
        g_max_idx[2*(i)]   = s_max_idx[0];
        g_max_idx[2*(i)+1] = s_max_idx[1];
    }
}

__global__ void clean2dc(int *dim1_p, int *dim2_p, int *argmax1_p, int *argmax2_p, float *stepr_p, \
                        float *stepi_p, float *ker, float *res, float *val_arr, float *g_nscore, \
                        float *g_max, int *g_max_idx){
    int dim1 = *dim1_p;
    int dim2 = *dim2_p;
    int argmax1 = *argmax1_p;
    int argmax2 = *argmax2_p;
    float stepr = *stepr_p;
    float stepi = *stepi_p;
    float valr, vali;
    extern __shared__ float s_nscore[];
    float *s_max   = s_nscore + blockDim.x*blockDim.y;
    int *s_max_idx = (int *)s_nscore + 3*blockDim.x*blockDim.y;
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    int n2 = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = n1 + blockDim.x*n2;
    s_max_idx[2*tid] = n1;
    s_max_idx[2*tid+1] = n2;
    if ((n1 < dim1) && (n2 < dim2)){
        int wrap_n1 = (n1 + argmax1) % dim1;
        int wrap_n2 = (n2 + argmax2) % dim2;
        res[2*(wrap_n1 + wrap_n2*dim1)]     -= (ker[2*(n1 + n2*dim1)] * stepr - ker[2*(n1 + n2*dim1)+1] * stepi);
        res[2*(wrap_n1 + wrap_n2*dim1) + 1] -= (ker[2*(n1 + n2*dim1)] * stepi + ker[2*(n1 + n2*dim1)+1] * stepr);
        valr = res[2*(wrap_n1 + wrap_n2*dim1)];
        vali = res[2*(wrap_n1 + wrap_n2*dim1) + 1];
        s_nscore[threadIdx.x + threadIdx.y*blockDim.x] = valr*valr+vali*vali;
        s_max[2*(threadIdx.x + threadIdx.y*blockDim.x)] = valr;
        s_max[2*(threadIdx.x + threadIdx.y*blockDim.x)+1] = vali;
    } else {
        s_nscore[tid] = 0;
        s_max[tid] = -1;
    }
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
    if (threadIdx.x == 0 && threadIdx.y == 0){
        g_nscore[blockIdx.x+blockDim.x*blockIdx.y]        = s_nscore[0];
        g_max[2*(blockIdx.x+blockDim.x*blockIdx.y)]       = s_max[0];
        g_max[2*(blockIdx.x+blockDim.x*blockIdx.y)+1]     = s_max[1];
        g_max_idx[2*(blockIdx.x+blockDim.x*blockIdx.y)]   = s_max_idx[0];
        g_max_idx[2*(blockIdx.x+blockDim.x*blockIdx.y)+1] = s_max_idx[1];
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
        int stop_if_div, \
        float *stepr_p, float *stepi_p, \
        int ker_len, int res_len, int dim1, int dim2,
        float *nscore_p, float *maxr_p, float *maxi_p, int *argmax1_p, int *argmax2_p) {
    int gridx, gridy;
    float *dev_ker, *dev_res, *dev_stepr, *dev_stepi, *dev_val_arr, *g_nscore, \
        *g_max;
    int *dev_argmax1, *dev_argmax2,
        *dev_dim1, *dev_dim2, *g_max_idx;
    float stepr = *stepr_p;
    float stepi = *stepi_p;
    int argmax1 = *argmax1_p;
    int argmax2 = *argmax2_p;
    float max_p[2];
    int max_idx_p[2];
    //Ceiling division of dim1/BLOCKSIZEX and dim2/BLOCKSIZEY
    gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
    gridy = (dim2 % BLOCKSIZEY == 0) ? dim2/BLOCKSIZEY : dim2/BLOCKSIZEY + 1;
    dim3 grid(gridx, gridy);
    dim3 blocksize(BLOCKSIZEX, BLOCKSIZEY);
    
    CudaSafeCall(cudaMalloc((void**) &dev_ker,      ker_len));
    CudaSafeCall(cudaMalloc((void**) &dev_res,      res_len));
    CudaSafeCall(cudaMalloc((void**) &dev_dim1,     sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_dim2,     sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_argmax1,  sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_argmax2,  sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &dev_stepr,    sizeof(float)));
    CudaSafeCall(cudaMalloc((void**) &dev_stepi,    sizeof(float)));
    CudaSafeCall(cudaMalloc((void**) &dev_val_arr,  2*sizeof(float)*dim1*dim2));
    CudaSafeCall(cudaMalloc((void**) &g_nscore,     sizeof(float)*gridx*gridy));
    CudaSafeCall(cudaMalloc((void**) &g_max,        2*sizeof(float)*gridx*gridy));
    CudaSafeCall(cudaMalloc((void**) &g_max_idx,    2*sizeof(int)*gridx*gridy));
    
    CudaSafeCall(cudaMemcpy(dev_ker,      ker,      ker_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_res,      res,      res_len,       cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_dim1,     &dim1,    sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_dim2,     &dim2,    sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_argmax1,  &argmax1, sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_argmax2,  &argmax2, sizeof(int),   cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_stepr,    &stepr,   sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_stepi,    &stepi,   sizeof(float), cudaMemcpyHostToDevice));
    
    int smemsize = 5*BLOCKSIZEX*BLOCKSIZEY*sizeof(float);
    // Take next step and compute score
    clean2dc<<<grid, blocksize, smemsize>>>(dev_dim1, dev_dim2, dev_argmax1, dev_argmax2, dev_stepr, \
                                dev_stepi, dev_ker, dev_res, dev_val_arr, g_nscore, g_max, g_max_idx);
    CudaSafeCall(cudaMemcpy(res, dev_res, res_len, cudaMemcpyDeviceToHost));
    CudaCheckError();
    
    int bsize = BLOCKSIZEX*BLOCKSIZEY;
    int gridsize = gridx*gridy/bsize;
    if ((gridsize % bsize) != 0 || gridsize == 0){
        gridsize += 1;
    }
    while(1){
        dim3 grid(gridsize, 1, 1);
        sum_max<<<grid, bsize, smemsize>>>(g_nscore, g_max, g_max_idx);
        CudaCheckError();
        if (gridsize % bsize == 0 && gridsize/bsize > 1){
            gridsize = gridsize/bsize;
        } else if (gridsize/bsize > 1){
            gridsize = gridsize/bsize +1;
        } else {
            dim3 grid(1,1,1);
            break;
        }
    }
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(nscore_p, g_nscore, sizeof(float), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(max_p, g_max, 2*sizeof(float), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(max_idx_p, g_max_idx, 2*sizeof(int), cudaMemcpyDeviceToHost));
    *argmax1_p = max_idx_p[0];
    *argmax2_p = max_idx_p[1];
    *maxr_p = max_p[0];
    *maxi_p = max_p[1];
    cudaFree(dev_ker);
    cudaFree(dev_res);
    cudaFree(dev_dim1);
    cudaFree(dev_dim2);
    cudaFree(dev_argmax1);
    cudaFree(dev_argmax2);
    cudaFree(dev_stepr);
    cudaFree(dev_stepi);
    cudaFree(dev_val_arr);
    cudaFree(g_nscore);
    cudaFree(g_max);
    cudaFree(g_max_idx);
    return 0;
}