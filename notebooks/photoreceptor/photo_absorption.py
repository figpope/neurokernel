import numpy as np
from math import exp,pow,factorial

class Photo_Receptor():
    def __init__(self):
	self.N_ph = 0
	self.N_micro = 30000
	self.N_m = self.N_micro
	self.lambdaM = 0.0
    def absorb(self, N_ph):
	self.N_ph = N_ph
	print self.N_ph, "photons"
	self.lambdaM = self.N_ph/self.N_m
	f_x = []
	k = 0
	while(1):
	    f =	pow(self.lambdaM,k)*exp(-self.lambdaM)
	    f =	float(f)/float(factorial(k))
	    k = k+1
	    if(f < 1/self.N_micro):
		break
	    else:
		f_x.append(f)
		print k,":",f




	

	
