% First absorb photons
time_step = 1000;
N_ph = 1000;
N_micro = 30000;
lambdaM = N_ph/N_micro;
k = 0;
output = zeros(time_step, N_micro);
p_k = [];
k_ind = [];
while true
    p = exp(-lambdaM)*power(lambdaM, k)/factorial(k);
    if p < 1/N_micro
        break;
    else
        p_k = [p_k, p];
        k_ind = [k_ind, k];
        k = k + 1;
    end
end
% 
num_activate = floor(N_micro*p_k);
for i = 1:time_step
    activate_1 = randsample(N_micro, sum(num_activate(2:3)));
    output(i, activate_1) = 1;
    activate_2 = randsample(length(activate_1), num_activate(3));
    output(i, activate_1(activate_2)) = 2;
end
%% 

