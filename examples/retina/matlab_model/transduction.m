%% Initialization 
X = [
  1, ...     M_star
  50, ...    G
  0, ...     G_star
  0, ...     PLC_star
  0, ...     D_star
  0, ...     C_star
  0        % T_star
];

% h
PLC_T = 100;
G_T = 50;
T_T = 25;
% Ca changes throughout the process, initially set to intracellular
% concentration in the dark 160nM
Ca = 160e-6;
% CaM assumed to be [C]i in the table
CaM = 0.5;

compute_h = @(X, Ca, CaM) [
  X(1), ...                                 M_star
  X(1)*X(2), ...                            M_star*G
  X(3)*(PLC_T-X(4)), ...                    G_star*(PLC_T-PLC_star)
  X(3)*X(4), ...                            G_star*PLC_star
  G_T-X(3)-X(2)-X(4), ...                   G_T-G_star-G-PLC_star
  X(4), ...                                 PLC_star
  X(4), ...                                 PLC_star
  X(5),...                                  D_star
  X(5)*(X(5)-1)*(T_T-X(7))/2,X(7), ...      D_star*(D_star-1)*(T_T-T_star)/2,T_star
  Ca*CaM, ...                               Ca*CaM
  X(6)                                    % C_star
];

h = compute_h(X, Ca, CaM);

% V
V_state_transition = [
  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0;
   0, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0;
   0,  1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0;
   0,  0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0;
   0,  0,  0,  0,  0,  1,  0, -1, -2,  0,  0,  0;
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1;
   0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0
];

% c
gamma_m_star = 3.7;
h_m_star = 40;

kappa_g_star = 7.05;
kappa_plc_star = 15.6;
gamma_gap = 3;
gamma_g = 3.5; % g_star?
kappa_d_star = 1300;
gamma_plc_star = 144;
h_plc_star = 11.1;

gamma_d_star = 4;
h_d_star = 37.8;

kappa_t_star = 150;
h_t_star_p = 11.5;

k_d_star = 1300;
gamma_t_star = 25;
h_t_star_n = 10;
K_mu = 30;
V = 3e-12;
K_R = 5.5;

K_P = 0.3;
K_N = 0.18;
m_p = 2;
m_n = 3;

n_s = 1;% 1 in dim background and 2 in bright background
calc_f_p = @(Ca) (Ca/K_P)^m_p/(1+(Ca/K_P)^m_p);
calc_f_n = @(C_star) n_s * (C_star/K_N)^m_n/(1+(C_star/K_N)^m_n);

f_p = calc_f_p(Ca);
f_n = calc_f_n(X(6));

c = [
  gamma_m_star*(1+h_m_star*f_n), ...
  kappa_g_star, ...
  kappa_plc_star, ...
  gamma_gap, ...
  gamma_g, ...
  kappa_d_star, ...
  gamma_plc_star*(1+h_plc_star*f_n), ...
  gamma_d_star*(1+h_d_star*f_n), ...
  kappa_t_star*(1+h_t_star_p*f_p)/(k_d_star^2), ...
  gamma_t_star*(1+h_t_star_n*f_n), ...
  K_mu/(V^2), ...
  K_R
];


%% Calcium Dynamic Constants
K_Na_Ca = 3e-8;
Na_o = 120;
Na_i = 8;
Ca_o = 1.5;
Ca_id = 160e-6; %Is this constant?
F = 96485;
V_m = -0.07;
R = 8.314;
T = 293;

n = 4; % binding sites
K_R = 5.5;
K_mu = 30;
K_Ca = 1000;

I_T_star = 0.68; % pA

calc_f1 = @(Na_i, Ca_o) K_Na_Ca * (Na_i^3*Ca_o) / (V*F);
calc_f2 = @(V_m, Na_o) K_Na_Ca * exp(-V_m*F/(R*T))*Na_o^3 / (V*F);
calc_Ca = @(C_star, CaM, I_Ca, f1, f2) V * (I_Ca/(2*V*F) + n*K_R*C_star - f1) / ...
                                           (n*K_mu*CaM + K_Ca - f2);
%% Start 
% Initialization
a = h.*c;
t = 0; 
t_end = 1;
N_ph = output(:,1);
T_ph = find(output(:,1))/1000;% in second
N_Rh = 0;
i = 1;
C_T = 0.5;
la = 0.5; % no fixed value, adjustable from 0.1~1
avo = 6.023e23;

X_out = X;
h_out = h;
% observe these
t_out = [];
mu_out = [];
I_in_out = [];
a_out = [];
while t < t_end
    r1 = rand();
    r2 = rand();

    a_mu = cumsum(a);
    
    a_s = sum(a);
    dt = 1/(la+a_s)*log(1/r1);
    
    if i <= length(T_ph) && t+dt > T_ph(i)
        t = T_ph(i);
        i = i+1;
        X(1) = X(1) + N_ph(i);
    else
        t = t+dt;
    end
    
    propensity = r2*a_s;
    found = false;
    for j=randperm(length(c))
        if a_mu(j) >= propensity && a_mu(j) ~= 0
            if j==1
                found = true;
                break;
            elseif a_mu(j-1) < propensity && a_mu(j) ~= 0
                found = true;
                break;
            end
        end
    end
    
    if found
        mu_out = [mu_out; j];

        % update X
        X = X + V_state_transition(:,j)';
        X_out = [X_out;X];
    end
        
    % update h
    C_star = (X(6)/avo)/(V*10^-3);
    CaM = C_T - C_star;
    h = compute_h(X, Ca, CaM);
    h_out = [h_out;h];

    % update c
    f_p = calc_f_p(Ca);
    f_n = calc_f_n(C_star);
    c(1) = gamma_m_star*(1+h_m_star*f_n);
    c(7) = gamma_plc_star*(1+h_plc_star*f_n);
    c(8) = gamma_d_star*(1+h_d_star*f_n);
    c(9) = kappa_t_star*(1+h_t_star_p*f_p)/(k_d_star^2);
    c(10) = gamma_t_star*(1+h_t_star_n*f_n);

    % update a
    a = h.*c;
    a_out = [a_out; a];

    % update Ca[2+] dynamics
    I_in = I_T_star*X(7);
    I_in_out = [I_in_out;I_in];
    I_Ca = 0.4*I_in;

    f1 = calc_f1(Na_i, Ca_o);
    f2 = calc_f2(V_m, Na_o);

    Ca = calc_Ca(C_star, CaM, I_Ca, f1, f2);

    t_out = [t_out;t];
end
