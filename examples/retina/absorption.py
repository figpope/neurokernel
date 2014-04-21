import numpy as np
def absorption(N_ph):
    time_step = 1000;
    #N_ph = 300;
    N_micro = 30000;
    lambdaM = float(N_ph)/N_micro;
    k = 0;
    output = np.zeros((time_step, N_micro),np.int64);
    for i in range(0,time_step-1):
	output[i,:] = np.random.poisson(lambdaM, N_micro)
    return output
