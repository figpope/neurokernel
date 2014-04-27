import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as garray
import curand as cr
import pycuda.driver as cuda
import atexit

# initialize device
cuda.init()
ctx = cuda.Device(0).make_context()
atexit.register(ctx.pop)

# create an example kernel that generate poisson distributed random numbers
template = """
#include "curand_kernel.h"

extern "C" {
__global__ void
gen_poisson_number(curandStateXORWOW_t *state, int* output, double lambda)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;

output[tid] = curand_poisson(&state[tid], lambda);
}
}
"""
mod = SourceModule(template, no_extern_c = True)
func = mod.get_function("gen_poisson_number")
func.prepare([np.intp, np.intp, np.double])


# generate N random numbers in GPU
N = 1000000
a = garray.empty(N, np.int32)
st = cr.curand_setup(N, 100)
# mean value of the poisson distribution.
lamb = 2

func.prepared_call( ((N-1)/256+1,1), (256,1,1), st.gpudata, a.gpudata, lamb)

print a.get().mean()



