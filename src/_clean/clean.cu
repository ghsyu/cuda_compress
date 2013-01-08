
/*
 * Some additional deconvolution functions for AIPY, written in C++ and CUDA.  These are
 * mostly for speed-critical applications. 
 *
 * Author: Aaron Parsons, Gilbert Hsyu
 */


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <Python.h>
#include "numpy/arrayobject.h"

#define QUOTE(s) # s

#define PNT1(a,i) (a->data + i*a->strides[0])
#define PNT2(a,i,j) (a->data+i*a->strides[0]+j*a->strides[1])
#define IND1(a,i,type) *((type *)PNT1(a,i))
#define IND2(a,i,j,type) *((type *)PNT2(a,i,j))
#define CIND1R(a,i,type) *((type *)PNT1(a,i))
#define CIND1I(a,i,type) *((type *)(PNT1(a,i)+sizeof(type)))
#define CIND2R(a,i,j,type) *((type *)PNT2(a,i,j))
#define CIND2I(a,i,j,type) *((type *)(PNT2(a,i,j)+sizeof(type)))

#define TYPE(a) a->descr->type_num
#define CHK_ARRAY_TYPE(a,type) \
    if (TYPE(a) != type) { \
        PyErr_Format(PyExc_ValueError, "type(%s) != %s", \
        QUOTE(a), QUOTE(type)); \
        return NULL; }

#define DIM(a,i) a->dimensions[i]
#define CHK_ARRAY_DIM(a,i,d) \
    if (DIM(a,i) != d) { \
        PyErr_Format(PyExc_ValueError, "dim(%s) != %s", \
        QUOTE(a), QUOTE(d)); \
        return NULL; }

#define RANK(a) a->nd
#define CHK_ARRAY_RANK(a,r) \
    if (RANK(a) != r) { \
        PyErr_Format(PyExc_ValueError, "rank(%s) != %s", \
        QUOTE(a), QUOTE(r)); \
        return NULL; }

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16




// A template for implementing addition loops for different data types
template<typename T> struct Clean {
    //   ____ ____  _   _ 
    //  / ___|  _ \| | | |
    // | |  _| |_) | | | |
    // | |_| |  __/| |_| |
    //  \____|_|    \___/ 
    __global__ void ker_gain(q_out, mq_out, T *ker, int kerd1, T *area){
        extern __shared__ float mq = 0;
        extern __shared__ float q = 0;
        n1 = threadIdx.x;
        n2 = threadIdx.y;
        val = ker[n1 + n2*kerd1];
        mval = val*val;
        //XXX Possible race condition?
        if (mval > mq && area(n1,n2)){
           mq = mval;
           q = val;
        }
        __syncthreads();
        if threadIdx.x = 0;{
            mq_out = mq;
            q_out = q
        }
    }                
    __global__ void clean2dr (int *dim1_p, int *dim2_p, int *argmax1_p, int *argmax2_p, float *step_p, T *ker, T *res,
                               int *nargmax1_p, int *nargmax2_p, T *max, T *mmax, int *pos_def, T *nscore){ 
        int dim1 = *dim1_p;
        int dim2 = *dim2_p;
        int argmax1 = *argmax1_p;
        int argmax2 = *argmax2_p;
        float step = *step_p;
        int nargmax1 = *nargmax1_p;
        int nargmax2 = *nargmax2_p;
        int gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
        T max=0, mmax, val, mval;
        //Array for accumulating the nscores of the block
        //Initialized to 0 for the case that there are more array elements than image pixels
        extern __shared__ T s_data[];
        int tid = threadIdx.x + BLOCKSIZEX*threadIdx.y;
        
        n1 = threadIdx.x + blockIdx.x * blockDim.x;
        n2 = threadIdx.y + blockIdx.y * blockDim.y;
        wrap_n1 = (n1 + argmax1) % dim1;
        wrap_n2 = (n2 + argmax2) % dim2;
        res[wrap_n1 + wrap_n2*size_of(T)*dim1] -= ker[n1 + n2*size_of(T)*dim1]*step;
        val = res[wrap_x + wrap_y*size_of(T)*dim1];
        mval = val * val;
        s_data[tid] = mval;
        for (int s = blockDim.x/2; s>0, s>>=1){
            if (tid < s){
                s_data[tid] += s_data[tid + s];
            }
        }
        __syncthreads();
        if (tid == 0){
            nscore[blockIdx.x + blockIdx.y*gridx] = s_data[0];
        //XXX Race condition?
        if (mval > mmax && (*pos_def == 0 || val > 0) && IND2(area,wrap_n1,wrap_n2,int)){
            nargmax1 = wrap_n1; nargmax2 = wrap_n2;
            max = val;
            mmax = mval;
        }
    }
    
    //   ____ _                  ____     _      
    //  / ___| | ___  __ _ _ __ |___ \ __| |_ __ 
    // | |   | |/ _ \/ _` | '_ \  __) / _` | '__|
    // | |___| |  __/ (_| | | | |/ __/ (_| | |   
    //  \____|_|\___|\__,_|_| |_|_____\__,_|_|   
    // Does a 2d real-valued clean
    static int clean_2d_r(PyArrayObject *res, PyArrayObject *ker,
            PyArrayObject *mdl, PyArrayObject *area, double gain, int maxiter, 
            double tol, int stop_if_div, int verb, int pos_def) {
        T score=-1, nscore, best_score=-1; 
        T max=0, mmax, val, mval, step, q=0, mq=0;
        T firstscore=-1;
        int argmax1=0, argmax2=0, nargmax1=0, nargmax2=0;
        int dim1=DIM(res,0), dim2=DIM(res,1), wrap_n1, wrap_n2;
        int gridx, gridy, smemsize;
        T *best_mdl=NULL, *best_res=NULL;
        T *dev_ker, *dev_area, *dev_res, *dev_max, *dev_mmax, *dev_step, *dev_mq, *dev_nscore
        int *dev_argmax1, *dev_argmax2, *dev_nargmax1, *dev_nargmax2,
            *dev_dim1, *dev_dim2
        if (!stop_if_div) {
            best_mdl = (T *)malloc(dim1*dim2*sizeof(T));
            best_res = (T *)malloc(dim1*dim2*sizeof(T));
        }
        // Compute gain/phase of kernel
        for (int n1=0; n1 < dim1; n1++) {
            for (int n2=0; n2 < dim2; n2++) {
                val = IND2(ker,n1,n2,T);
                mval = val * val;
                if (mval > mq && IND2(area,n1,n2,int)) {
                    mq = mval;
                    q = val;
                }
            }
        }
        q = 1/q;
        cudaMalloc((void**) &dev_ker,      PyArray_NBYTES(ker));
        cudaMalloc((void**) &dev_res,      PyArray_NBYTES(res));
        cudaMalloc((void**) &dev_dim1,     sizeof(int));
        cudaMalloc((void**) &dev_dim2,     sizeof(int));
        cudaMalloc((void**) &dev_argmax1,  sizeof(int));
        cudaMalloc((void**) &dev_argmax2,  sizeof(int));
        cudaMalloc((void**) &dev_step,     sizeof(T));
        cudaMalloc((void**) &dev_nargmax1, sizeof(int));
        cudaMalloc((void**) &dev_nargmax2, sizeof(int));
        cudaMalloc((void**) &dev_max,      sizeof(T));
        cudaMalloc((void**) &dev_mmax,     sizeof(T));
        cudaMalloc((void**) &dev_pos_def,  sizeof(int));
        cudaMalloc((void**) &dev_nscore,   sizeof(T)*(dim1*dim2/(BLOCKSIZEX * BLOCKSIZEY)+1));
        
        cudaMemCpy(dev_ker,      PyArray_DATA(ker), PyArray_NBYTES(ker),    cudaMemcpyHostToDevice);
        cudaMemCpy(dev_res,      PyArray_DATA(res), PyArray_NBYTES(res),    cudaMemcpyHostToDevice);
        cudaMemCpy(dev_dim1,     &dim1,             sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemCpy(dev_dim2,     &dim2,             sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemCpy(dev_argmax1,  &argmax1,          sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemCpy(dev_argmax2,  &argmax2,          sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemCpy(dev_step,     &step,             sizeof(T),              cudaMemcpyHostToDevice);
        cudaMemCpy(dev_nargmax1, &nargmax1,         sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemCpy(dev_nargmax2, &nargmax2,         sizeof(int),            cudaMemcpyHostToDevice);
        cudaMemCpy(dev_max,      &max,              sizeof(T),              cudaMemcpyHostToDevice);
        cudaMemCpy(dev_pos_def,  &pos_def,          sizeof(int),            cudaMemcpyHostToDevice);
        //Ceiling division of dim1/16 and dim2/16
        gridx = (dim1 % BLOCKSIZEX == 0) ? dim1/BLOCKSIZEX : dim1/BLOCKSIZEX + 1;
        gridy = (dim2 % BLOCKSIZEY == 0) ? dim2/BLOCKSIZEY : dim2/BLOCKSIZEY + 1;
        dim3 grid(gridx, gridy);
        dim3 blocksize(BLOCKSIZEX, BLOCKSIZEY);
        smemsize = BLOCKSIZEX * BLOCKSIZEY;
        // The clean loop
        for (int i=0; i < maxiter; i++) {
            nscore = 0;
            mmax = -1;
            cudaMemCpy(dev_mmax, &mmax, sizeof(T), cudaMemcpyHostToDevice);
            step = (T) gain * max * q;
            IND2(mdl,argmax1,argmax2,T) += step;
            // Take next step and compute score
            //XXX
            clean2dr<<<grid, blocksize,>>>(dev_dim1, dev_dim2, dev_argmax1, dev_argmax2, dev_step, dev_ker,
                                      dev_res, dev_nargmax1, dev_nargmax2, dev_max, dev_mmax, dev_pos_def, dev_nscore);
            
            cudaMemCpy(&max,   dev_max,  sizeof(T), cudaMemcpyDeviceToHost);
            
            nscore = sqrt(nscore / (dim1 * dim2));
            if (firstscore < 0) firstscore = nscore;
            if (verb != 0)
                printf("Iter %d: Max=(%d,%d,%f), Score=%f, Prev=%f, Delta=%f\n", \
                    i, nargmax1, nargmax2, max, (double) (nscore/firstscore), \
                    (double) (score/firstscore), 
                    (double) fabs(score - nscore) / firstscore);
            if (score > 0 && nscore > score) {
                if (stop_if_div) {
                    // We've diverged: undo last step and give up
                    IND2(mdl,argmax1,argmax2,T) -= step;
                    for (int n1=0; n1 < dim1; n1++) {
                        wrap_n1 = (n1 + argmax1) % dim1;
                        for (int n2=0; n2 < dim2; n2++) {
                            wrap_n2 = (n2 + argmax2) % dim2;
                            IND2(res,wrap_n1,wrap_n2,T) += IND2(ker,n1,n2,T) * step;
                        }
                    }
                    return -i;
                } else if (best_score < 0 || score < best_score) {
                    // We've diverged: buf prev score in case it's global best
                    for (int n1=0; n1 < dim1; n1++) {
                        wrap_n1 = (n1 + argmax1) % dim1;
                        for (int n2=0; n2 < dim2; n2++) {
                            wrap_n2 = (n2 + argmax2) % dim2;
                            best_mdl[n1*dim1+n2] = IND2(mdl,n1,n2,T);
                            best_res[wrap_n1*dim1+wrap_n2] = IND2(res,wrap_n1,wrap_n2,T) + IND2(ker,n1,n2,T) * step;
                        }
                    }
                    best_mdl[argmax1*dim1+argmax2] -= step;
                    best_score = score;
                    i = 0;  // Reset maxiter counter
                }
            } else if (score > 0 && fabs(score - nscore) / firstscore < tol) {
                // We're done
                if (best_mdl != NULL) { free(best_mdl); free(best_res); }
                return i;
            } else if (not stop_if_div && (best_score < 0 || nscore < best_score)) {
                i = 0;  // Reset maxiter counter
            }
            score = nscore;
            argmax1 = nargmax1; argmax2 = nargmax2;
        }
        // If we end on maxiter, then make sure mdl/res reflect best score
        if (best_score > 0 && best_score < nscore) {
            for (int n1=0; n1 < dim1; n1++) {
                for (int n2=0; n2 < dim2; n2++) {
                    IND2(mdl,n1,n2,T) = best_mdl[n1*dim1+n2];
                    IND2(res,n1,n2,T) = best_res[n1*dim1+n2];
                }
            }
        }   
        if (best_mdl != NULL) { free(best_mdl); free(best_res); }
        cudaFree(dev_ker);
        cudaFree(dev_res);
        cudaFree(dev_dim1);
        cudaFree(dev_dim2);
        cudaFree(dev_argmax1);
        cudaFree(dev_argmax2);
        cudaFree(dev_step);
        cudaFree(dev_nargmax1);
        cudaFree(dev_nargmax2);
        cudaFree(dev_max);
        cudaFree(dev_mmax);
        cudaFree(pos_def);
        cudaFree(max_nscore);
        return maxiter;
    }
}