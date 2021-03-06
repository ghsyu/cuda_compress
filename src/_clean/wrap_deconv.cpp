/*
 * Some additional deconvolution functions for AIPY, written in C++.  These are
 * mostly for speed-critical applications. 
 *
 * Author: Aaron Parsons
 */

#include <Python.h>
#include "numpy/arrayobject.h"
#include "deconv.h"

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


static int clean_2d_c(PyArrayObject *res, PyArrayObject *ker,
        PyArrayObject *mdl, PyArrayObject *area, double gain, int maxiter, double tol,
        int stop_if_div, int verb, int pos_def) {
    float maxr=0, maxi=0, valr, vali, stepr, stepi, qr=0, qi=0;
    float score=-1, nscore, best_score=-1;
    float mmax, mval, mq=0;
    float firstscore=-1;
    int argmax1=0, argmax2=0, nargmax1=0, nargmax2=0;
    int dim1=DIM(res,0), dim2=DIM(res,1), wrap_n1, wrap_n2;
    float *best_mdl=NULL, *best_res=NULL;
    float *dev_ker, *dev_res, *g_nscore_i, *g_max_i, *g_nscore_o, *g_max_o;
    int *dev_area, *g_max_idx_i, *g_max_idx_o;
    if (!stop_if_div) {
        best_mdl = (float *)malloc(2*dim1*dim2*sizeof(float));
        best_res = (float *)malloc(2*dim1*dim2*sizeof(float));
    }
    // Compute gain/phase of kernel
    for (int n1=0; n1 < dim1; n1++) {
        for (int n2=0; n2 < dim2; n2++) {
            valr = CIND2R(ker,n1,n2,float);
            vali = CIND2I(ker,n1,n2,float);
            mval = valr * valr + vali * vali;
            if (mval > mq && IND2(area,n1,n2,int)) {
                mq = mval;
                qr = valr; qi = vali;
            }
        }
    }
    qr /= mq;
    qi = -qi / mq;
    //Malloc arrays on GPU
    gpu_set_up(&dev_ker, &dev_res, &dev_area,       \
               &g_nscore_i, &g_max_i, &g_max_idx_i, \
               &g_nscore_o, &g_max_o, &g_max_idx_o, \
               (float *)PyArray_DATA(ker), (float *)PyArray_DATA(res), (int *)PyArray_DATA(area), \
                dim1, dim2, PyArray_NBYTES(ker), PyArray_NBYTES(res), PyArray_NBYTES(area));
    for (int i=0; i < maxiter; i++) {
        nscore = 0;
        mmax = -1;
        stepr = (float) gain * (maxr * qr - maxi * qi);
        stepi = (float) gain * (maxr * qi + maxi * qr);
        CIND2R(mdl,argmax1,argmax2,float) += stepr;
        CIND2I(mdl,argmax1,argmax2,float) += stepi;
        // Take next step and compute score
        clean_2d_c_GPU((float *)PyArray_DATA(res), (float *)PyArray_DATA(ker), (int *)PyArray_DATA(area), \
                    gain, maxiter, stop_if_div, stepr, stepi, argmax1, argmax2, \
                    PyArray_NBYTES(ker), PyArray_NBYTES(res), PyArray_NBYTES(area), dim1, dim2, \
                    &nscore, &maxr, &maxi, &nargmax1, &nargmax2, \
                    dev_ker, dev_res, dev_area, \
                    g_nscore_i, g_max_i, g_max_idx_i, g_nscore_o, g_max_o, g_max_idx_o);
        nscore = sqrt(nscore / (dim1 * dim2));
        if (firstscore < 0) firstscore = nscore;
        if (verb != 0)
            printf("Iter %d: Max=(%d,%d), Score = %f, Prev = %f\n", \
                i, nargmax1, nargmax2, (double) (nscore/firstscore), \
                (double) (score/firstscore));
        if (score > 0 && nscore > score) {
            if (stop_if_div) {
                // We've diverged: undo last step and give up
                CIND2R(mdl,argmax1,argmax2,float) -= stepr;
                CIND2I(mdl,argmax1,argmax2,float) -= stepi;
                for (int n1=0; n1 < dim1; n1++) {
                    wrap_n1 = (n1 + argmax1) % dim1;
                    for (int n2=0; n2 < dim2; n2++) {
                        wrap_n2 = (n2 + argmax2) % dim2;
                        CIND2R(res,wrap_n1,wrap_n2,float) += CIND2R(ker,n1,n2,float)*stepr - CIND2I(ker,n1,n2,float)*stepi;
                        CIND2I(res,wrap_n1,wrap_n2,float) += CIND2R(ker,n1,n2,float)*stepi + CIND2I(ker,n1,n2,float)*stepr;
                    }
                }
                return -i;
            } else if (best_score < 0 || score < best_score) {
                // We've diverged: buf prev score in case it's global best
                for (int n1=0; n1 < dim1; n1++) {
                    wrap_n1 = (n1 + argmax1) % dim1;
                    for (int n2=0; n2 < dim2; n2++) {
                        wrap_n2 = (n2 + argmax2) % dim2;
                        best_mdl[2*(n1*dim2+n2)+0] = CIND2R(mdl,n1,n2,float);
                        best_mdl[2*(n1*dim2+n2)+1] = CIND2I(mdl,n1,n2,float);
                        best_res[2*(wrap_n1*dim2+wrap_n2)+0] = CIND2R(res,wrap_n1,wrap_n2,float) + CIND2R(ker,n1,n2,float) * stepr - CIND2I(ker,n1,n2,float) * stepi;
                        best_res[2*(wrap_n1*dim2+wrap_n2)+1] = CIND2I(res,wrap_n1,wrap_n2,float) + CIND2R(ker,n1,n2,float) * stepi + CIND2I(ker,n1,n2,float) * stepr;
                    }
                }
                best_mdl[2*(argmax1*dim2+argmax2)+0] -= stepr;
                best_mdl[2*(argmax1*dim2+argmax2)+1] -= stepi;
                best_score = score;
                i = 0;  // Reset maxiter counter
            }
        } else if (score > 0 && (score - nscore) / firstscore < tol) {
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
                CIND2R(mdl,n1,n2,float) = best_mdl[2*(n1*dim2+n2)+0];
                CIND2I(mdl,n1,n2,float) = best_mdl[2*(n1*dim2+n2)+1];
                CIND2R(res,n1,n2,float) = best_res[2*(n1*dim2+n2)+0];
                CIND2I(res,n1,n2,float) = best_res[2*(n1*dim2+n2)+1];
            }
        }
    }
    if (best_mdl != NULL) { free(best_mdl); free(best_res); }
    //Free GPU arrays
    gpu_free(dev_ker, dev_res, dev_area, g_nscore_i, g_max_i, g_max_idx_i, g_nscore_o, g_max_o, g_max_idx_o);
    return maxiter;
}
// __        __                               
// \ \      / / __ __ _ _ __  _ __   ___ _ __ 
//  \ \ /\ / / '__/ _` | '_ \| '_ \ / _ \ '__|
//   \ V  V /| | | (_| | |_) | |_) |  __/ |   
//    \_/\_/ |_|  \__,_| .__/| .__/ \___|_|   
//                     |_|   |_|              

// Clean wrapper that handles all different data types and dimensions
PyObject *clean(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *res, *ker, *mdl, *area;
    double gain=.1, tol=.001;
    int maxiter=200, rank=0, dim1, dim2, rv, stop_if_div=0, verb=0, pos_def=0;
    static char *kwlist[] = {"res", "ker", "mdl", "area", "gain", \
                             "maxiter", "tol", "stop_if_div", "verbose","pos_def", NULL};
    // Parse arguments and perform sanity check
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!|didiii", kwlist, \
            &PyArray_Type, &res, &PyArray_Type, &ker, &PyArray_Type, &mdl, &PyArray_Type, &area, 
            &gain, &maxiter, &tol, &stop_if_div, &verb, &pos_def)) 
        return NULL;
    if (RANK(res) == 1) {
        rank = 1;
        CHK_ARRAY_RANK(ker, 1); CHK_ARRAY_RANK(mdl, 1); CHK_ARRAY_RANK(area, 1);
        dim1 = DIM(res,0);
        CHK_ARRAY_DIM(ker, 0, dim1); CHK_ARRAY_DIM(mdl, 0, dim1); CHK_ARRAY_DIM(area, 0, dim1);
    } else if (RANK(res) == 2) {
        rank = 2;
        CHK_ARRAY_RANK(ker, 2); CHK_ARRAY_RANK(mdl, 2); CHK_ARRAY_RANK(area, 2);
        dim1 = DIM(res,0); dim2 = DIM(res,1);
        CHK_ARRAY_DIM(ker, 0, dim1); CHK_ARRAY_DIM(mdl, 0, dim1); CHK_ARRAY_DIM(area, 0, dim1);
        CHK_ARRAY_DIM(ker, 1, dim2); CHK_ARRAY_DIM(mdl, 1, dim2); CHK_ARRAY_DIM(area, 1, dim2);
    }
    if (TYPE(res) != TYPE(ker) || TYPE(res) != TYPE(mdl)) {
        PyErr_Format(PyExc_ValueError, "array types must match");
        return NULL;
    }
    if (!(PyArray_ISCARRAY(res) && PyArray_ISONESEGMENT(res))) {
        PyErr_Format(PyExc_ValueError, "array must be aligned and one segment");
        return NULL;
    }
    if (!(PyArray_ISCARRAY(ker) && PyArray_ISONESEGMENT(ker))) {
        PyErr_Format(PyExc_ValueError, "Kernel array must be aligned and one segment");
        return NULL;
    }
    if (TYPE(area) != NPY_LONG) {
        PyErr_Format(PyExc_ValueError, "area must be of type 'int'");
        return NULL;
    }
    Py_INCREF(res); Py_INCREF(ker); Py_INCREF(mdl);
    // Use template to implement data loops for all data types

    if (TYPE(res) == NPY_CFLOAT && rank == 2) {
            rv = clean_2d_c(res,ker,mdl,area,gain,maxiter,tol,stop_if_div,verb,pos_def);
    } else {
        PyErr_Format(PyExc_ValueError, "Unsupported data type.");
        return NULL;
    }
    Py_DECREF(res); Py_DECREF(ker); Py_DECREF(mdl);
    return Py_BuildValue("i", rv);
}

// Wrap function into module
static PyMethodDef DeconvGPUMethods[] = {
    {"clean", (PyCFunction)clean, METH_VARARGS|METH_KEYWORDS,
        "clean(res,ker,mdl,gain=.1,maxiter=200,tol=.001,stop_if_div=0,verbose=0,pos_def=0)\nPerform a 1 or 2 dimensional deconvolution using the CLEAN algorithm.."},
    {NULL, NULL}
};

PyMODINIT_FUNC init_deconvGPU(void) {
    (void) Py_InitModule("_deconvGPU", DeconvGPUMethods);
    import_array();
};
