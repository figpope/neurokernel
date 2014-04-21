# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from absorption import absorption

N_ph = 300
photons = absorption(N_ph)
print photons

mod = SourceModule("""
#include<stdio.h>

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
__device__ void find_T_ph(float* T_ph, int* final_T, float* N_ph){
  int j = 0;
  for(int i = 0; i < 1000; i++){
    if(N_ph[i] != 0){
      T_ph[j] = N_ph[i] / 1000;
      j++;
    }
  }
  *final_T = j-1;
}
__global__ void transduction(float *dest, float *X_out, float *src, float* photons)
{
  float Ca = 160e-6;
  float CaM = 0.5;
  float V_m = -0.07;
  
  const int i = threadIdx.x;
  
  float X[7] = {1, 50, 0, 0, 0, 0, 0};
  float h[12];
  
  compute_h(&h, X, Ca, CaM, PLC_T, G_T, T_T);

  float a[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  
  float r1 = 0.5;
  float r2 = 0.5;
  
  float f_p = calc_f_p(Ca);
  float f_n = calc_f_n(X[5]);
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
  float *N_ph = photons; 
  
  float *T_ph = new float[1000];
  int final_T = 0;
  find_T_ph(T_ph, &final_T, N_ph);

  float propensity;
  int found, index, ind_i = 0;
  
  while(t < t_end){
    r1 = 0.5;
    r2 = 0.5;

    for(int j = 0; j < 12; j++){
      a[j] = h[j]*c[j];
    }

    cumsum(a_mu, a);
    a_s = a_mu[11];

    dt = 1/(la+a_s)*logf(1/r1);
    
    if(ind_i <= final_T && t+dt > T_ph[ind_i]){
        t = T_ph[ind_i];
        index = 1000*T_ph[ind_i];
        X[0] = X[0] + N_ph[index];
        ind_i++;
    }
    else { t = t+dt };
    
    propensity = r2*a_s;
    found = 0;
    for(int j = 0; j < 12; j++){
      if(a_mu[j] >= propensity && a_mu[j] != 0){
        if(j == 1) {
          found = 1;
          break;
        } else if(a_mu[j-1] < propensity) {
          found = 1;
          break;
        }
      }
    }

    if(found){
      for(int k = 0; k < 7; k++){
        X[k] += V_state_transition[k][j];
      }
      step++;
    }
    
    C_star = (X[5]/avo)/(V*powf(10,-3));
    CaM = C_T - C_star;
    compute_h(h, X, Ca, CaM);
    f_p = calc_f_p(Ca);
    f_n = calc_f_n(C_star);
    
    // To be optimized
    c[0] = gamma_m_star*(1+h_m_star*f_n);
    c[6] = gamma_plc_star*(1+h_plc_star*f_n);
    c[7] = gamma_d_star*(1+h_d_star*f_n);
    c[8] = kappa_t_star*(1+h_t_star_p*f_p)/powf(k_d_star,2);
    c[9] = gamma_t_star*(1+h_t_star_n*f_n);
    
    for(int j = 0; j < 12; j++){
        a[j] = h[j] * c[j];
    }

    float I_in = I_T_star*X[6];
    float I_Ca = 0.4*I_in;

    float Ca = calc_Ca(C_star, CaM, I_Ca, V_m);     
  }
}
""", options = ["--ptxas-options=-v"])

transduction = mod.get_function("transduction")
testin = np.zeros((100,1),np.float32)
dest = np.zeros_like(testin)
d = np.ones(2,np.int32)
grid = (int(d[0]),int(d[1]))
print grid
photon_array = np.array(photons[:,0])
print photon_array
X_out = np.zeros((10000*7,1), np.float32)
transduction( drv.Out(dest), drv.Out(X_out), drv.In(testin),drv.In(photon_array),
                    block=(100,1,1), grid=grid)
#print dest
X_out = np.reshape(X_out, (10000,7))
print X_out


# <codecell>


