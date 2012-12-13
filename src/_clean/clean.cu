#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void ker_gain(dim1, dim2){
     n1 = threadIdx.x;
     n2 = threadIdx.y;
     //XXX
     val = ker(n1, n2);
     mval = val*val;
     if (mval > mq && area(n1,n2)){
        mq = mval;
        q = val;
     }
}

__global__ void clean2dr (int dim1, int dim2, int argmax1, int argmax2, float step, float *ker, float *res){
    n1 = threadIdx.x;
    n2 = threadIdx.y;
    wrap_n1 = (n1 + argmax1) % dim1;
    wrap_n2 = (n2 + argmax)2 % dim2;
    //XXX 
    res(wrap_n1, wrap_n2) -= ker (x,y)*step;
    val = res(wrap_x, wrap_y);
    mval = val * val;
    nscore += mval;
    if (mval > mq && area(wrap_n1, wrap_n2)){
        nargmax1 = wrap_n1; nargmax2 = wrap_n2;
        max = val;
        mmax = mval;
    }
}


int clean2dr(){
    float score = -1, nscore, best_score = -1;
    float max = 0, mmax, val, mval, step, q=0, mq-0;
    float firstscore = -1;
    int argmax1=0, argmax2=0, nargmax1=0, nargmax2=0;
        int dim1=DIM(res,0), dim2=DIM(res,1), wrap_n1, wrap_n2;
        float *best_mdl=NULL, *best_res=NULL;
        //XXX move these over to GPU?
        if (!stop_if_div) {
            best_mdl = (float *)malloc(dim1*dim2*sizeof(float));
            best_res = (float *)malloc(dim1*dim2*sizeof(float));
        }
    cudaMalloc2D(,)
}