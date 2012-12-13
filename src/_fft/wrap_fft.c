#include <Python.h>
#include <cufft.h>
#include "comfft.h"
#include <cuda_runtime_api.h>
#include "numpy/arrayobject.h"

PyObject *wrap_fft2d(PyObject *self, PyObject *args){
	PyArrayObject *src;
    PyArrayObject *dst;
	int nx;
	int ny;
    
	if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &src)){
	return NULL;
	}
	
    if (PyArray_NDIM((PyObject *)src) != 2){
        PyErr_Format(PyExc_ValueError, "src data must be 2 dimensional");
        return NULL;
    }
	else{
		nx = PyArray_DIM(src, 0);
		ny = PyArray_DIM(src, 1);
	}
	
	if (! (PyArray_ISCOMPLEX((PyObject *) src))){
        PyErr_Format(PyExc_ValueError, "All arrays must be complex");
        return NULL;
	}
	
	dst = (PyArrayObject *) PyArray_SimpleNew(2, PyArray_DIMS(src), NPY_CFLOAT);

	fft2d((cufftComplex *)PyArray_DATA(src), (cufftComplex *)PyArray_DATA(dst), nx, ny);

	return PyArray_Return(dst);
	}

static PyMethodDef cudafftMethods[] = {
	{"fft2d", (PyCFunction)wrap_fft2d, METH_VARARGS,
		"fft2d (src data) Does a 2D fft on the input data using a GPU"
        },
	{NULL, NULL}
	};
	
PyMODINIT_FUNC init_fft(void){
	(void) Py_InitModule("_fft", cudafftMethods);
	import_array();
};
