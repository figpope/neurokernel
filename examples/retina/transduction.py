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
__device__ void compute_h(float* h, float* X, float Ca, float CaM, const float PLC_T, const float G_T, const float T_T){
    /*float h[12] = {X[0],
                  X[0]*X[1],                             
                  X[2]*(PLC_T-X[3]),                    
                  X[2]*X[3],                             
                  G_T-X[2]-X[1]-X[3],                    
                  X[3],                                  
                  X[3],                                 
                  X[4],                                  
                  X[4]*(X[4]-1)*(T_T-X[6])/2,           
                  X[6],                  
                  Ca*CaM,
                  X[5]
                  };
    */
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
    float result = powf(Ca/K_P, m_p)/(1+powf(Ca/K_P, m_p));
    return result;
}
__device__ float calc_f_n(float C_star, float K_N, float m_n, float n_s){
    float result = n_s*powf(C_star/K_N, m_n)/(1+powf(C_star/K_N, m_n));
    return result;
}
__device__ void cumsum(float* a_mu, float* a){
    int i;
    a_mu[0] = a[0];
    for(i=1;i<12;i++){
	a_mu[i] = a_mu[i-1] + a[i];
    }
}
__device__ void find_T_ph(float* T_ph, int* final_T, float* N_ph){
    int i,j;
    j=0;
    for(i=0;i<1000;i++){
	if(N_ph[i]!=0){
	    T_ph[j]=N_ph[i]/1000;
	    j++;
	}
    }
    *final_T = j-1;
}
__global__ void transduction(float *dest, float *X_out, float *src, float* photons)
{
    const float PLC_T = 100;
    const float G_T = 50;
    const float T_T = 25;
    float Ca = 160e-6;
    float CaM = 0.5;
    const float gamma_m_star = 3.7;
    const float h_m_star = 40;

    const float kappa_g_star = 7.05;
    const float kappa_plc_star = 15.6;
    const float gamma_gap = 3;
    const float gamma_g = 3.5;

    const float kappa_d_star = 1300;
    const float gamma_plc_star = 144;
    const float h_plc_star = 11.1;

    const float gamma_d_star = 4;
    const float h_d_star = 37.8;

    const float kappa_t_star = 150;
    const float h_t_star_p = 11.5;

    const float k_d_star = 1300;
    const float gamma_t_star = 25;
    const float h_t_star_n = 10;
    const float K_mu = 30;
    const float V = 3e-12;
    const float K_R = 5.5;

    const float K_P = 0.3;
    const float K_N = 0.18;
    const float m_p = 2;
    const float m_n = 3;

    const float n_s = 1;
    const float K_Na_Ca = 3e-8;
    const float Na_o = 120;
    const float Na_i = 8;
    const float Ca_o = 1.5;
    const float Ca_id = 160e-6;
    const float F = 96485;
    const float V_m = -70;
    const float R = 8.314;
    const float T = 293;

    const float n = 4;
    const float K_Ca = 1000;

    const float I_T_star = 0.68;
    const float C_T = 0.5;
    const float la = 0.5;
    
    const int i = threadIdx.x;
    
    float X[7] = {1,50,0,0,0,0,0};
    //float h[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    float h[12] = {X[0],
                  X[0]*X[1],                             
                  X[2]*(PLC_T-X[3]),                    
                  X[2]*X[3],                             
                  G_T-X[2]-X[1]-X[3],                    
                  X[3],                                  
                  X[3],                                 
                  X[4],                                  
                  X[4]*(X[4]-1)*(T_T-X[6])/2,           
                  X[6],                  
                  Ca*CaM,
                  X[5]
                  };
    //compute_h(&h, X, Ca, CaM, PLC_T, G_T, T_T);
    
    float V_state_transition[7][12] = {
    {-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
   {0, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
   {0,  1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0},
   {0,  0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0},
   {0,  0,  0,  0,  0,  1,  0, -1, -2,  0,  0,  0},
   {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1},
   {0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0}};
    
    float f_p = calc_f_p(Ca, K_P, m_p);
    float f_n = calc_f_n(X[6], K_N, m_n, n_s);
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

    float a[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    int j = 0;
    for(j=0;j<12;j++){
        a[j] = h[j]*c[j];
    }
    float t = 0;
    float t_end = 1;
    float r1 = 0.5;
    float r2 = 0.5;
    float a_mu[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    float a_s = a_mu[11];
    float dt = 1/(la+a_s)*logf(1/r1);
    float *N_ph = photons; 
    
    float *T_ph = new float[1000];
    int final_T = 0;
    find_T_ph(T_ph, &final_T, N_ph);

    float propensity = 0;
    int found = 0;
    int index = 0;
    int ind_i = 0;
    int k = 0;
    int step = 0;
    while(t<t_end){
	r1 = 0.5;
	r2 = 0.5;
	cumsum(a_mu, a);
	a_s = a_mu[11];
	dt = 1/(la+a_s)*logf(1/r1);
	if(ind_i<=final_T && t+dt > T_ph[ind_i]){
	    t = T_ph[ind_i];
	    index = 1000*T_ph[ind_i];
	    X[0] = X[0] + N_ph[index];
	    //X[0] = 1;
	    ind_i++;
	}
	else
	    t = t+dt;
	
	propensity = r2*a_s;
	found = 0;
	for(j=0;j<12;j++){
	    if(a_mu[j] >= propensity && a_mu[j] != 0){
		if(j==1){
		    found = 1;
		    break;
		}
		else if(a_mu[j-1] < propensity && a_mu[j] != 0){
		    found = 1;
		    break;
		}
	    }
	}

	if(found){
	    for(k=0;k<7;k++){
		X[k] += V_state_transition[k][j];
		X_out[step*7+k] = X[k];
	    }
	    step++;
	}
	   
	CaM = C_T - X[6];
	compute_h(h, X, Ca, CaM, PLC_T, G_T, T_T);
	f_p = calc_f_p(Ca, K_P, m_p);
	f_n = calc_f_n(X[6], K_N, m_n, n_s);
	c[1] = gamma_m_star*(1+h_m_star*f_n);
	c[7] = gamma_plc_star*(1+h_plc_star*f_n);
        c[8] = gamma_d_star*(1+h_d_star*f_n);
        c[9] = kappa_t_star*(1+h_t_star_p*f_p)/powf(k_d_star,2);
        c[10] = gamma_t_star*(1+h_t_star_n*f_n);
	for(j=0;j<12;j++){
	    a[j] = h[j]*c[j];
	}

	//calcium dynamics


	

        
    }


     
    
    delete(T_ph);
    dest[i] = src[i] + 1.0;
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


