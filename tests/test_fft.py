import aipy
import cuda_compress._fft as f
import numpy as n, unittest

def test_zeroes():
    src_data = n.zeros((100,100), dtype=n.complex)
    gpu = f.fft2d(src_data)
    cpu = n.fft.fft2(src_data)
    print(gpu)
    print(cpu)
    print(gpu-cpu)
    
if __name__ == '__main__':
    test_zeroes()
    