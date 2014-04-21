from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from neurokernel.LPU.utils.curand import curand_setup

NUM_MICROVILLI = 30000

class Photoreceptor(BaseNeuron):
    def __init__(self, n_dict, V, dt , debug=False, LPU_id=None):
        super(Photoreceptor, self).__init__(n_dict, V, dt, debug, LPU_id)
        self.num_neurons = len(n_dict['id'])
        self.debug = debug

        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)),1)
        self.ddt = dt / self.steps

        self.V = V
        
        self.sa  = garray.to_gpu(np.asarray(n_dict['sa'], dtype=np.float64))
        self.si  = garray.to_gpu(np.asarray(n_dict['si'], dtype=np.float64))
        self.dra = garray.to_gpu(np.asarray(n_dict['dra'], dtype=np.float64))
        self.dri = garray.to_gpu(np.asarray(n_dict['dri'], dtype=np.float64))

        X_init = [0,50,0,0,0,0,0]
        self.X = garray.to_gpu(np.asarray([[X_init for i in range(NUM_MICROVILLI)] for neuron in range(self.num_neurons)], dtype=np.int))
        self.Ca = garray.to_gpu(np.asarray([[0 for i in range(NUM_MICROVILLI)] for neuron in range(self.num_neurons)], dtype=np.float64))

        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['initV'], dtype=np.double))

        self.state = curand_setup(self.num_neurons*NUM_MICROVILLI,100)

        self.lam = garray.to_gpu(np.asarray(np.zeros([30000, self.num_neurons], dtype=np.double)))

        self.update_microvilli = self.get_microvilli_kernel()
        self.update_hhn = self.get_hhn_kernel()
    
    @property
    def neuron_class(self): return True

    def eval(self, st = None):
        self.update_microvilli.prepared_async_call(self.update_grid, self.update_block, st,
                                                   self.num_neurons, self.state.gpudata, self.lam.gpudata,
                                                   self.X.gpudata, self.Ca.gpudata, self.ddt*1000,
                                                   self.I.gpudata, self.V.gpudata)

    def get_hhn_kernel(self):
        template = """
    #define NNEU %(nneu)d //NROW * NCOL

    #define E_K -85    // potassium reversal potential
    #define E_Cl -30   // chloride reversal potential
    #define G_s 1.6    // maximum shaker conductance
    #define G_dr 3.5   // maximum delayed rectifier conductance
    #define G_Cl 0.056 // chloride leak conductance
    #define G_K 0.082  // potassium leak conductance
    #define C 4        // membrane capacitance

    __global__ void
    hhn_model(%(type)s* V, %(type)s* sa, %(type)s* si, %(type)s* dra, %(type)s* dri, \ 
                       int num_neurons, %(type)s* I_pre, %(type)s dt) {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;

        if(cart_id < num_neurons) {
            // computing voltage gated time constants and steady-state
            // activation/inactivation functions
            sa_inf = (1./(1+exp((-30-V)/13.5))).^(1/3);
            tau_sa = 0.13+3.39*exp(-(-73-V).^2./20^2);
            si_inf = 1./(1+exp((-55-V)/-5.5));
            tau_si = 113*exp(-(-71-V).^2./29^2);
            dra_inf = (1./(1+exp((-5-V)/9))).^(1/2);
            tau_dra = 0.5+5.75*exp(-(-25-V).^2./32^2);
            dri_inf = 1./(1+exp((-25-V)/-10.5));
            tau_dri = 890;

            // compute derivatives
            dsa = (sa_inf - sa[cart_id])./tau_sa;
            dsi = (si_inf - si[cart_id])./tau_si;
            ddra = (dra_inf - dra[cart_id])./tau_dra;
            ddri = (dri_inf - dri[cart_id])./tau_dri;
            dV = (I_pre[cart_id] - G_K*(V[cart_id]-E_K) - G_Cl * (V[cart_id]-E_Cl) - G_s * sa * si * (V[cart_id]-E_K) - G_dr * dra * dri * (V[cart_id]-E_K) - 0.093*(V[cart_id]-10) )/C;

            V   += dt*dV;
            sa  += dt*sa;
            si  += dt*si;
            dra += dt*ddra;
            dri += dt*ddri;
        }
    }
    """#Used 40 registers, 1024+0 bytes smem, 84 bytes cmem[0], 308 bytes cmem[2], 28 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128,1,1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),  "nneu": self.update_block[0]}, options=["--ptxas-options=-v"])
        func = mod.get_function("hhn_model")

        func.prepare([np.intp, 
                      np.intp,
                      np.intp,
                      np.intp,
                      np.intp,
                      np.intp,
                      np.int32,
                      np.intp,
                      np.intp])

        return func

    def get_microvilli_kernel(self):
        template = """
    #include<stdio.h>

    #define NNEU %(nneu)d //NROW * NCOL

    #define PLC_T 100
    #define G_T 50
    #define T_T 25

    #define kappa_g_star 7.05
    #define kappa_plc_star 15.6
    #define kappa_d_star 1300
    #define kappa_t_star 150

    #define gamma_gap 3
    #define gamma_g 3.5
    #define gamma_plc_star 144
    #define gamma_m_star 3.7
    #define gamma_d_star 4
    #define gamma_t_star 25

    #define h_plc_star 11.1
    #define h_d_star 37.8
    #define h_t_star_p 11.5
    #define h_t_star_n 10
    #define h_m_star 40

    #define k_d_star 1300
    #define K_mu 30
    #define V 3e-12
    #define K_R 5.5
    #define K_P 0.3
    #define K_N 0.18
    #define m_p 2
    #define m_n 3
    #define n_s 1
    #define K_Na_Ca 3e-8
    #define Na_o 120
    #define Na_i 8
    #define Ca_o 1.5
    #define Ca_id 160e-6
    #define F 96485
    #define R 8.314
    #define T 293
    #define n 4
    #define K_Ca 1000
    #define I_T_star 0.68
    #define C_T 0.5

    #define avo 6.023e23
    #define NUM_MICROVILLI 30000

    #define la = 0.5

    __constant__ int V_state_transition[7][12] = {
      {-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      { 0, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
      { 0,  1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0},
      { 0,  0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0},
      { 0,  0,  0,  0,  0,  1,  0, -1, -2,  0,  0,  0},
      { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1},
      { 0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0}
    };

    __device__ void compute_h(float* h, float* X, float Ca, float CaM){
      h[0] = X[0];
      h[1] = X[0]*X[1];                             
      h[2] = X[2]*(PLC_T-X[3]);                    
      h[3] = X[2]*X[3];                             
      h[4] = G_T-X[2]-X[1]-X[3];                    
      h[5] = X[3];                                  
      h[6] = X[3];                                 
      h[7] = X[4];                                  
      h[8] = X[4]*(X[4]-1)*(T_T-X[6])/2;           
      h[9] = X[6];                  
      h[10] = Ca*CaM;
      h[11] = X[5];
    }

    __device__ float calc_f_p(float Ca, float K_P, float m_p){
      return powf(Ca/K_P, m_p)/(1+powf(Ca/K_P, m_p));
    }

    __device__ float calc_f_n(float C_star){
      return n_s*powf(C_star/K_N, m_n)/(1+powf(C_star/K_N, m_n));
    }

    __device__ float calc_f1(float Na_i, float Ca_o) {
      return K_Na_Ca * (powf(Na_i, 3)*Ca_o) / (V*F);
    }

    __device__ float calc_f2(float V_m, float Na_o) {
      return K_Na_Ca * exp(-V_m*F/(R*T))*powf(Na_o,3) / (V*F);
    }

    __device__ float calc_Ca(float C_star, float CaM, float I_Ca, float V_m) {
      return V * (I_Ca/(2*V*F) + n*K_R*C_star - calc_f1(Na_i, Ca_o)) / (n*K_mu*CaM + K_Ca - calc_f2(V_m, Na_o));
    }

    __device__ void cumsum(float* out, float* a){
      out[0] = a[0];
      for(int i = 1; i < 12; i++){
        a_mu[i] = a_mu[i-1] + a[i];
      }
    }

    __global__ void transduction(int num_neurons, curandStateXORWOW_t *state, %(type)s* lambda, \
                                 %(type)s* X, %(type)s* Ca, %(type)s* dt, %(type)s* I, %(type)s* V_m)
    {
      int tid = blockIdx.x * NNEU + threadIdx.x;
      int mid = tid % num_microvilli;
      int nid = tid / num_microvilli;
      
      if(nid < num_neurons * num_microvilli) {
        X[nid][mid][0] += curand_poisson(&state[tid], lambda)
        float a[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

        C_star = (X[5]/avo)/(V*powf(10,-3));
        CaM = C_T - C_star;
        
        float h[12];
        compute_h(&h, X[nid][mid], Ca[nid][tid], CaM);

        float r1 = curand_uniform(&state[tid]);
        float r2 = curand_uniform(&state[tid]);
        
        float f_p = calc_f_p(Ca);
        float f_n = calc_f_n(C_star);
        float c[12] = {
          gamma_m_star*(1+h_m_star*f_n),
          kappa_g_star,
          kappa_plc_star,
          gamma_gap,
          gamma_g,
          kappa_d_star,
          gamma_plc_star*(1+h_plc_star*f_n),
          gamma_d_star*(1+h_d_star*f_n),
          kappa_t_star*(1+h_t_star_p*f_p)/(powf(k_d_star,2)),
          gamma_t_star*(1+h_t_star_n*f_n),
          K_mu/powf(V,2),
          K_R
        };

        float a_mu[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
        
        int found = 0;

        for(int j = 0; j < 12; j++){
          a[j] = h[j]*c[j];
        }

        cumsum(a_mu, a);
        a_s = a_mu[11];

        dt = 1/(la+a_s)*logf(1/r1);
        
        float propensity = r2*a_s;
        for(int j = 0; j < 12; j++){
          if(a_mu[j] >= propensity && a_mu[j] != 0){
            if(j == 1) {
              found = 1; break;
            } else if(a_mu[j-1] < propensity) {
              found = 1; break;
            }
          }
        }

        if(found){
          for(int k = 0; k < 7; k++){
            X[nid][mid][k] += V_state_transition[k][j];
          }
        }

        I[nid] = I_T_star*X[tid][mid][6];
        float I_Ca = 0.4*I;

        // update C_star
        C_star = (X[5]/avo)/(V*powf(10,-3));
        CaM = C_T - C_star;

        Ca[tid][mid] = calc_Ca(C_star, CaM, I_Ca, V_m[tid]);
      }
    }
    """ # Used 29 registers, 104 bytes cmem[0], 56 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128,1,1)
        self.update_grid = ((self._num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),  "nneu": self.update_block[0]}, options=["--ptxas-options=-v"])
        func = mod.get_function("transduction")

        func.prepare([np.int,     # num_neurons
                      np.intp,    # state
                      np.intp,    # lambda
                      np.intp,    # X
                      np.intp,    # Ca
                      scalartype, # dt
                      np.intp,    # I
                      np.intp])   # V_m

        return func
