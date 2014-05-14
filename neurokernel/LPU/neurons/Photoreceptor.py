from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from neurokernel.LPU.utils.curand import curand_setup

import tables
from jinja2 import Template

NUM_MICROVILLI = 1000

class Photoreceptor(BaseNeuron):
    def __init__(self, n_dict, V, dt , debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.LPU_id = None
        super(Photoreceptor, self).__init__(n_dict, V, dt, debug, LPU_id)
        self.debug = debug

        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)),1)
        self.ddt = dt / self.steps

        self.V = V
        
        self.sa  = garray.to_gpu(np.asarray(n_dict['sa'], dtype=np.double))
        self.si  = garray.to_gpu(np.asarray(n_dict['si'], dtype=np.double))
        self.dra = garray.to_gpu(np.asarray(n_dict['dra'], dtype=np.double))
        self.dri = garray.to_gpu(np.asarray(n_dict['dri'], dtype=np.double))

        X_init = [0,50,0,0,0,0,0]
        self.X = garray.to_gpu(np.asarray([[X_init for i in range(NUM_MICROVILLI)] for neuron in range(self.num_neurons)], dtype=np.int))

        self.I_micro = garray.to_gpu(np.asarray(np.zeros([self.num_neurons, NUM_MICROVILLI], dtype=np.double)))

        self.dt_micro = garray.to_gpu(np.asarray(np.zeros([self.num_neurons, NUM_MICROVILLI], dtype=np.double)))

        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['initV'], dtype=np.double))

        self.state = curand_setup(self.num_neurons*NUM_MICROVILLI,100)

        self.update_microvilli = self.get_microvilli_kernel()
        self.update_hhn = self.get_hhn_kernel()

        if self.debug:
            if self.LPU_id is None:
                self.LPU_id = "anon"
            self.I_file = tables.openFile(self.LPU_id + "_I.h5", mode="w")
            self.I_file.createEArray("/","array", \
                                     tables.Float64Atom(), (0,self.num_neurons))
    
    @property
    def neuron_class(self): return True

    def eval(self, st = None):
        self.update_microvilli.prepared_async_call(self.micro_update_grid, self.micro_update_block, st,
                                                   self.num_neurons, self.state.gpudata, self.I.gpudata, 
                                                   self.X.gpudata, self.ddt*10, self.I_micro.gpudata, self.V, self.dt_micro.gpudata)
        self.update_hhn.prepared_async_call(self.hhn_update_grid, self.hhn_update_block, st,
                                            self.num_neurons, self.ddt*1000, self.V, self.sa.gpudata, 
                                            self.si.gpudata, self.dra.gpudata, self.dri.gpudata, 
                                            self.I_micro.gpudata, self.I.gpudata)
        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1,-1)))

    def get_hhn_kernel(self):
        template = """
  #define NNEU %(nneu)d //NROW * NCOL
  #define NUM_MICROVILLI %(num_micro)d

  #define E_K (-85)      // potassium reversal potential
  #define E_Cl (-30)     // chloride reversal potential
  #define G_s 1.6        // maximum shaker conductance
  #define G_dr 3.5       // maximum delayed rectifier conductance
  #define G_Cl 0.056     // chloride leak conductance
  #define G_K 0.082      // potassium leak conductance
  #define C 4            // membrane capacitance
  #define m_V (1.57e-5)  // membrane volume

  __global__ void hhn_model(int num_neurons, %(type)s dt, %(type)s* V, %(type)s* sa, %(type)s* si, \
                   %(type)s* dra, %(type)s* dri, %(type)s (*I_micro)[NUM_MICROVILLI], %(type)s* I) {
    int nid = blockIdx.x * NNEU + threadIdx.x;

    if(nid < num_neurons) {
        V[nid] *= 1000;
        I[nid] = 0;
        for(int i = 0; i < NUM_MICROVILLI; i++) {
          I[nid] += I_micro[nid][i];
        }
        float I_pre = I[nid] / m_V;
      
  /*
        // computing voltage gated time constants and steady-state
        // activation/inactivation functions
        float sa_inf = powf(1 / (1 + expf((-30 - V[nid]) / 13.5)), 1/3);
        float tau_sa = 0.13 + 3.39 * exp(powf(-(-73 - V[nid]), 2) / 400);
        float si_inf = 1 / (1 + expf((-55 - V[nid]) / -5.5));
        float tau_si = powf(113 * expf(-(-71 - V[nid])), 2) / 841;
        float dra_inf = powf(1 / (1 + expf((-5 - V[nid]) / 9)), 1/2);
        float tau_dra = 0.5 + 5.75 * exp(powf(-(-25 - V[nid]), 2) / 1024);
        float dri_inf = 1 / (1 + expf((-25 - V[nid]) / -10.5));
        float tau_dri = 890;

        // compute derivatives
        float dsa = (sa_inf - sa[nid])/tau_sa;
        float dsi = (si_inf - si[nid])/tau_si;
        float ddra = (dra_inf - dra[nid])/tau_dra;
        float ddri = (dri_inf - dri[nid])/tau_dri;
        float dV = (I_pre - G_K * (V[nid] - E_K) \
                          - G_Cl * (V[nid] - E_Cl) \
                          - G_s * sa[nid] * si[nid] * (V[nid] - E_K) \
                          - G_dr * dra[nid] * dri[nid] * (V[nid] - E_K) \
                          - 0.093 * (V[nid] - 10)) \
                    / C;

        V[nid]   += dt*dV;
        sa[nid]  += dt*dsa;
        si[nid]  += dt*dsi;
        dra[nid] += dt*ddra;
        dri[nid] += dt*ddri;
        
        V[nid] /= 10000;
  */
  V[nid] = I[nid];
    }
  }
    """# Used 41 registers, 96 bytes cmem[0], 56 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.hhn_update_block = (128,1,1)
        self.hhn_update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),  "nneu": self.hhn_update_block[0], 'num_micro': NUM_MICROVILLI}, options=["--ptxas-options=-v"])
        func = mod.get_function("hhn_model")

        func.prepare([np.int32,       # num_neurons
                      scalartype,     # dt
                      np.intp,        # V
                      np.intp,        # Sa
                      np.intp,        # Si
                      np.intp,        # Dra
                      np.intp,        # Dri
                      np.intp,        # I_micro
                      np.intp])       # I

        return func

    def get_microvilli_kernel(self):
        template = Template("""
    #include "curand_kernel.h"

    extern "C" {
        #define NNEU {{ nneu }} //NROW * NCOL
    
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
        #define V 3e-9
        #define K_R 5.5
        #define K_P 0.3
        #define K_N 0.18
        #define m_p 2
        #define m_n 3
        #define n_s 2
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
        #define C_T 903.45
    
        #define avo 6.023e23
        #define NUM_MICROVILLI {{ num_micro }}
    
        #define TRP_rev 0.013
    
        #define la 0.5
        #define concentration_ratio 1806.9
    
        __constant__ int V_state_transition[7][12] = {
          {-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
          { 0, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
          { 0,  1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0},
          { 0,  0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0},
          { 0,  0,  0,  0,  0,  1,  0, -1, -2,  0,  0,  0},
          { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1},
          { 0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0}
        };
        
        __device__ float calc_f_p(float Ca){
          return powf(Ca/K_P, m_p)/(1+powf(Ca/K_P, m_p));
        }
    
        __device__ float calc_f_n(float C_star){
          return n_s*powf(C_star/K_N, m_n)/(1+powf(C_star/K_N, m_n));
        }
    
        __device__ float calc_f1() {
          return K_Na_Ca * (powf(Na_i, 3)*Ca_o) / (V*F);
        }
    
        __device__ float calc_f2(float V_m) {
          return K_Na_Ca * exp(-V_m*F/(R*T))*powf(Na_o,3) / (V*F);
        }
    
        __device__ float calc_Ca(float C_star, float CaM, float I_Ca, float V_m) {
          return (I_Ca/(2*V*F) + n*K_R*C_star + calc_f1()) / (n*K_mu*CaM + K_Ca + calc_f2(V_m));
        }
    
        __device__ void cumsum(double* a_mu, double* a){
          a_mu[0] = a[0];
          for(int i = 1; i < 12; i++){
            a_mu[i] = a_mu[i-1] + a[i];
          }
        }
    
        __global__ void transduction(int num_neurons, curandStateXORWOW_t *state, {{ type }}* photon_input, \
                                     int (*X)[NUM_MICROVILLI][7], {{ type }} ddt, {{ type }} (*I_micro)[NUM_MICROVILLI], \
                                     {{ type }}* V_m, {{type}} (*dt_micro)[NUM_MICROVILLI]) {
          int tid = blockIdx.x * NNEU + threadIdx.x;
          int mid = tid % NUM_MICROVILLI;
          int nid = tid / NUM_MICROVILLI;

          if (tid < num_neurons * NUM_MICROVILLI) {
            float t = 0;
            double a_mu[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            double a_s = 0;
            double a[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            double propensity = 0;
            float C_star_conc, CaM, CaM_conc, Ca, I_Ca, f_p, f_n, r1, r2;

            X[nid][mid][0] += curand_poisson(&state[tid], photon_input[nid] / NUM_MICROVILLI);
    
            int steps = 0;
            while(t + dt_micro[nid][mid] < ddt && steps < 2000){
              r1 = curand_uniform(&state[tid]);
              r2 = curand_uniform(&state[tid]);
              
              C_star_conc = X[nid][mid][5] / concentration_ratio;
              CaM = C_T - X[nid][mid][5];
              CaM_conc = CaM / concentration_ratio;

              if(TRP_rev > V_m[nid])
                I_micro[nid][mid] = X[nid][mid][6] * 8 * (TRP_rev - V_m[nid]);
              else
                I_micro[nid][mid] = 0;
              
              I_Ca = 0.4 * I_micro[nid][mid];
              Ca = calc_Ca(C_star_conc, CaM_conc, I_Ca, V_m[nid]);

              f_p = calc_f_p(Ca);
              f_n = calc_f_n(C_star_conc);
    
              a[0] = X[nid][mid][0] * gamma_m_star * (1 + h_m_star * f_n);
              a[1] = X[nid][mid][0] * X[nid][mid][1] * kappa_g_star;
              a[2] = X[nid][mid][2] * (PLC_T - X[nid][mid][3]) * kappa_plc_star;
              a[3] = X[nid][mid][2] * X[nid][mid][3] * gamma_gap;
              a[4] = (G_T - X[nid][mid][2] - X[nid][mid][1] - X[nid][mid][3]) * gamma_g;
              a[5] = X[nid][mid][3] * kappa_d_star;
              a[6] = X[nid][mid][3] * gamma_plc_star * (1 + h_plc_star * f_n);
              a[7] = X[nid][mid][4] * gamma_d_star * (1 + h_d_star * f_n);
              a[8] = (X[nid][mid][4] * (X[nid][mid][4] - 1) * (T_T - X[nid][mid][6]) / 2) * (kappa_t_star * (1 + h_t_star_p * f_p) / (powf(k_d_star, 2)));
              a[9] = X[nid][mid][6] * gamma_t_star * (1 + h_t_star_n * f_n);
              a[10] = Ca * CaM * K_mu;
              a[11] = X[nid][mid][5] * K_R;

              cumsum(a_mu, a);
              a_s = a_mu[11];
              if(dt_micro[nid][mid] == 0) {
                dt_micro[nid][mid] = 1/(la+a_s)*logf(1/r1);
              }
                    
              if(t + dt_micro[nid][mid] < ddt){
                propensity = r2 * a_s;
                int j, found = 0;
                for (j = 0; j < 12; j++) {
                  if (a_mu[j] >= propensity && a_mu[j] != 0) {
                    if (j == 0) {
                      found = 1; break;
                    } else if (a_mu[j - 1] < propensity) {
                      found = 1; break;
                    }
                  }
                }
        
                if (found) {
                  for (int k = 0; k < 7; k++) {
                    X[nid][mid][k] += V_state_transition[k][j];
                  }
                }
                t += dt_micro[nid][mid];
                dt_micro[nid][mid] = 0;
              }
              steps++;
            }
          dt_micro[nid][mid] -= ddt - t;
          }
        }
      }
    """) # Used 29 registers, 104 bytes cmem[0], 56 bytes cmem[16]
        dtype = np.double
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.micro_update_block = (512,1,1)
        self.micro_update_grid = (((NUM_MICROVILLI - 1) / self.micro_update_block[0] + 1) * self.num_neurons, 1)
        mod = SourceModule(template.render(type=dtype_to_ctype(dtype), nneu=self.micro_update_block[0], num_micro=NUM_MICROVILLI), options=["--ptxas-options=-v"], no_extern_c=True)
        func = mod.get_function("transduction")

        func.prepare([np.int32,   # num_neurons
                      np.intp,    # state
                      np.intp,    # lambda
                      np.intp,    # X
                      scalartype, # ddt
                      np.intp,    # I_micro
                      np.intp,    # V_m
                      np.intp])   # dt_micro

        return func
