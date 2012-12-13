import aipy
import cuda_compress._fft as f
import numpy as n, unittest

class Test_fft(object):
    def __init__(self):
        pass
    def set_zeroes(self, shape = (128,128)):
        self.src_data = n.zeros(shape, dtype = n.complex64)
    def set_ones(self, shape = (128,128)):
        self.src_data = n.ones(shape, dtype = n.complex64)
    def set_gaussian(self, size = 128, fwhm = 60):
        i = n.arange(0, size, 1, n.complex64)
        j = i[:,n.newaxis]
        i0 = j0 = size //2
        self.src_data = n.exp(-4*n.log(2) * ((i-i0)**2 + (j-j0)**2) / fwhm**2)
    def set_random(self, shape = (128,128)):
        self.src_data = n.random.random(shape).astype(n.complex64)
    def set_one(self, shape = (128,128)):
        self.src_data = n.zeros(shape, dtype = n.complex64)
        self.src_data[shape[0]//2, shape[1]//2] = 1
    def set_1d_sine(self, size = 128):
        self.src_data = n.array([n.sin(n.arange(size,dtype=n.complex64))]*size)
    def run(self):
        gpu = f.fft2d(self.src_data)
        cpu = n.fft.fft2(self.src_data)
        if n.all(gpu-cpu) == 0:
            print 'Test success'
        else:
            print gpu
    
if __name__ == '__main__':
    A = Test_fft()
    A.set_one()
    A.run()
