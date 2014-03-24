from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class HodgkinHuxley(BaseNeuron):
    def __init__(self, n_dict, spk, dt , debug=False, LPU_id=None):
        super(HodgkinHuxley, self).__init__(n_dict, spk, dt, debug, LPU_id)
        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug

        self.spk = spk

        self.V       = garray.to_gpu(np.asarray(n_dict['V'],       dtype=np.float64))
        self.V_prev  = garray.to_gpu(np.asarray(n_dict['V_prev'],  dtype=np.float64))
        self.X_1     = garray.to_gpu(np.asarray(n_dict['X_1'],     dtype=np.float64))
        self.X_2     = garray.to_gpu(np.asarray(n_dict['X_2'],     dtype=np.float64))
        self.X_3     = garray.to_gpu(np.asarray(n_dict['X_3'],     dtype=np.float64))
        self.update = self.get_kernel()
    
    def post_run(self):
        print self.I
        print self.V
        print self.V_prev

    @property
    def neuron_class(self): return True

    def eval(self, st = None):
        self.update.prepared_async_call(self.update_grid, self.update_block, st, self.spk, 
                                        self._num_neurons, self.I.gpudata, self.dt, 
                                        self.X_1.gpudata, self.X_2.gpudata, self.X_3.gpudata, 
                                        self.V.gpudata, self.V_prev.gpudata)

    def get_kernel(self):
        template = """
    #define NNEU %(nneu)d //NROW * NCOL

    #define g_Na 120.0
    #define g_K  36.0
    #define g_L  0.3
    #define E_K  (-12.0)
    #define E_Na 115.0
    #define E_L  10.613

    __global__ void
    hhn_model(int *spk, int num_neurons, %(type)s* I_pre, %(type)s dt, \
              %(type)s* X_1, %(type)s* X_2, %(type)s* X_3, %(type)s* g_V, %(type)s* V_prev)
    {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;

        if(cart_id < num_neurons)
        {
            %(type)s V = g_V[cart_id];
            spk[cart_id] = 0;

            %(type)s a[3];

            a[0] = (10-V)/(100*(exp((10-V)/10)-1));
            X_1[cart_id] = a[0]*dt - X_1[cart_id]*(dt*(a[0] + exp(-V/80)/8) - 1);
           
            a[1] = (25-V)/(10*(exp((25-V)/10)-1));
            X_2[cart_id] = a[1]*dt - X_2[cart_id]*(dt*(a[1] + 4*exp(-V/18)) - 1);
           
            a[2] = 0.07*exp(-V/20);
            X_3[cart_id] = a[2]*dt - X_3[cart_id]*(dt*(a[2] + 1/(exp((30-V)/10)+1)) - 1);

            V = V + dt * (I_pre[cart_id] - \
               (g_K * pow(X_1[cart_id], 4) * (V - E_K) + \
                g_Na * pow(X_2[cart_id], 3) * X_3[cart_id] * (V - E_Na) + \
                g_L * (V - E_L)));

            if(V_prev[cart_id] <= g_V[cart_id] && g_V[cart_id] > V) {
                spk[cart_id] = 1;
            }
            
            V_prev[cart_id] = g_V[cart_id];
            g_V[cart_id] = V;
        }
    }
    """ # Used 29 registers, 104 bytes cmem[0], 56 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128,1,1)
        self.update_grid = ((self._num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),  "nneu": self.update_block[0]}, options=["--ptxas-options=-v"])
        func = mod.get_function("hhn_model")

        func.prepare([np.intp,      # spk array
                      np.int32,     # num_neurons
                      np.intp,      # I_pre
                      scalartype,   # dt
                      np.intp,      # X_1
                      np.intp,      # X_2
                      np.intp,      # X_3
                      np.intp,      # g_V
                      np.intp])     # V_prev

        return func
