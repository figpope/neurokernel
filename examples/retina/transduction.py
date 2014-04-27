
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from absorption import absorption

N_ph = 100
photons = absorption(N_ph)
#print photons

mod = SourceModule("""
#include<stdio.h>

#define avo 6.023e23
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
#define C_T 903.45


#define la 0.5
#define V_m -0.07 

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
__device__ float calc_f_p(float Ca){
  return powf(Ca/K_P, m_p)/(1+powf(Ca/K_P, m_p));
}

__device__ float calc_f_n(float C_star){
  return n_s*powf(C_star/K_N, m_n)/(1+powf(C_star/K_N, m_n));
}

__device__ float calc_f1() {
  return K_Na_Ca * (powf(Na_i, 3)*Ca_o) / (V*F);
}

__device__ float calc_f2() {
  return K_Na_Ca * exp(-V_m*F/(R*T))*powf(Na_o,3) / (V*F);
}

__device__ float calc_Ca(float C_star, float CaM, float I_Ca) {
  return (I_Ca/(2*V*F) + n*K_R*C_star + calc_f1()) / (n*K_mu*CaM + K_Ca + calc_f2());
}

__device__ void cumsum(float* a_mu, float* a){
  a_mu[0] = a[0];
  for(int i = 1; i < 12; i++){
    a_mu[i] = a_mu[i-1] + a[i];
  }
}
__device__ void find_T_ph(float* T_ph, int* final_T, float* N_ph){
  int j = 0;
  for(int i = 0; i < 1000; i++){
    if(N_ph[i] != 0){
      T_ph[j] =  (float)i / 1000;
      *final_T = j;
      j++;
    }
  }
  printf("device final_T %d\\n", *final_T);
}
__global__ void transduction(float *dest, float *X_out, float *src, float* photons, float* rand_array)
{
  float Ca = 160e-6;
  float CaM = C_T;
  
  const int i = threadIdx.x;
  
  float X[7] = {1, 50, 0, 0, 0, 0, 0};
  float h[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  
  compute_h(h, X, Ca, CaM);

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
    K_mu,
    K_R
  };
  


  float a_mu[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  float *N_ph = photons; 
  
  float *T_ph = new float[1000];
  int final_T = 0;
  
  //for(int j=0;j<1000;j++){
  //  printf("%f \\n",N_ph[j]);
  //}
  find_T_ph(T_ph, &final_T, N_ph);
  printf("final_T: %d\\n", final_T);
  for(int j=0;j<final_T;j++)
    printf("%f ", T_ph[j]);

  float propensity;
  int found, index, ind_i = 0;
  float C_star = 0;
  float C_star_concentration = 0;
  float CaM_concentration = 0.5;

  float t = 0;
  float t_end = 1;
  float a_s =  0;
  float dt = 0;
  int step = 0;

  //choose X, a, h, 
  char watching = 'n';

  
  while(t < t_end){
    //r1 = 0.5;
    //r2 = 0.5;
    r1 = rand_array[step*2];
    r2 = rand_array[step*2+1];
    //printf("%f \\n", rand_array[step]);

    for(int j = 0; j < 12; j++){
      a[j] = h[j]*c[j];
      if(watching == 'a')
	printf("%f ", a[j]);
    }
    
    if(watching == 'a')
	printf("\\n");

    cumsum(a_mu, a);
    a_s = a_mu[11];

    dt = 1/(la+a_s)*logf(1/r1);
    
    if(ind_i <= final_T && t+dt > T_ph[ind_i]){
        t = T_ph[ind_i];
        index = 1000*T_ph[ind_i];
        X[0] = X[0] + N_ph[index];
        ind_i++;
    }
    else { t = t+dt; };
    
    propensity = r2*a_s;
    found = 0;
    int j = 0;
    for(j = 0; j < 12; j++){
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
	X_out[step*7+k] = X[k];
	if(watching == 'X')
	    printf("%f ", X_out[step*7+k]);
      }
      if(watching == 'X')
	printf("\\n");
      step++;
    }
    
    C_star = X[5];
    C_star_concentration = C_star/avo*1e3/(V*1e-9);
    CaM = C_T - C_star;
    CaM_concentration = CaM/avo*1e3/(V*1e-9);
    compute_h(h, X, Ca, CaM_concentration);
    f_p = calc_f_p(Ca);
    f_n = calc_f_n(C_star_concentration);
    
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

    float Ca = calc_Ca(C_star_concentration, CaM_concentration, I_Ca);     
  }
}
""", options = ["--ptxas-options=-v"])

transduction = mod.get_function("transduction")
testin = np.zeros((100,1),np.float32)
dest = np.zeros_like(testin)
d = np.ones(2,np.int32)
grid = (int(d[0]),int(d[1]))
#print grid
photon_array = np.array(photons[:,0],np.float32)
#print photon_array
rand_array = np.random.rand(100000);
rand_array =  np.array(rand_array, np.float32)
X_out = np.zeros((10000*7,1), np.float32)
transduction( drv.Out(dest), drv.Out(X_out), drv.In(testin),drv.In(photon_array),drv.In(rand_array),
                    block=(1,1,1), grid=grid)
#print dest
X_out = np.reshape(X_out, (10000,7))
#print X_out




