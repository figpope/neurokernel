%Absorption
N_photoreceptor = 2;
N_micro = 10;
% lambdaM = N_ph/N_micro;
% photons = poissrnd(lambdaM, 1, 1000)
load('working.mat') % preloaded data
photon_input = retina_input(1,:);

%% Initialization 
X = cell(N_photoreceptor, N_micro);
[X{:}] = deal([ 
  1, ...     M_star
  50, ...    G
  0, ...     G_star
  0, ...     PLC_star
  0, ...     D_star
  0, ...     C_star
  0        % T_star
]);

% X = [
%   1, ...     M_star
%   50, ...    G
%   0, ...     G_star
%   0, ...     PLC_star
%   0, ...     D_star
%   0, ...     C_star
%   0        % T_star
% ];

% h
avo = 6.023e23;
PLC_T = 100;
G_T = 50;
T_T = 25;
% Ca changes throughout the process, initially set to intracellular
% concentration in the dark 160nM
Ca = cell(N_photoreceptor, N_micro);
[Ca{:}] = deal(160e-6);
% CaM assumed to be [C]i in the table
%V = 3e-12;%V uL
V = 3e-9;
Concentration_ratio = avo*1e-3*(V*1e-9);
CaM = cell(N_photoreceptor, N_micro);
[CaM{:}] = deal(0.5*Concentration_ratio);

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

h = compute_h(X{1,1}, Ca{1,1}, CaM{1,1});

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

K_R = 5.5;

K_P = 0.3;
K_N = 0.18;
m_p = 2;
m_n = 3;

n_s = 2;% 1 in dim background and 2 in bright background
calc_f_p = @(Ca) (Ca/K_P)^m_p/(1+(Ca/K_P)^m_p);
calc_f_n = @(C_star) n_s * (C_star/K_N)^m_n/(1+(C_star/K_N)^m_n);

f_p = calc_f_p(Ca{1});
f_n = calc_f_n(X{1}(6));
c = cell(N_photoreceptor, N_micro);
[c{:}] = deal([
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
  K_mu, ...
  K_R
]);

%% Calcium Dynamic Constants
K_Na_Ca = 3e-8;
Na_o = 120;
Na_i = 8;
Ca_o = 1.5;
Ca_id = 160e-6; %Is this constant?
F = 96485;
%V_m = -70;
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
calc_Ca = @(C_star, CaM, I_Ca, f1, f2) (I_Ca/(2*V*F) + n*K_R*C_star + f1) / ...
                                           (n*K_mu*CaM + K_Ca + f2);
%% Start 
% Initialization
a = cell(N_photoreceptor, N_micro);
[a{:}] = deal(h.*c{1,1});
t = 0; 
timestep = 1e-4;
t_end = t+timestep;
t_terminate = 1;
% N_ph = photons;
% T_ph = find(photons)/1000;% in second
N_Rh = 0;
i = 1;
C_T = 0.5*Concentration_ratio;
la = 0.5; % no fixed value, adjustable from 0.1~1

X_out = X;
h_out = h;
% observe these
t_out = [];
mu_out = [];
%I_in_out = [];
a_out = [];
C_star_out = [];
CaM_out = [];
Ca_out = [];

%% Iteration
%for each microvillus
X_out = [];
iteration_number = t_terminate / timestep;
I_in_output = cell(N_photoreceptor, N_micro);
[I_in_output{:}] = deal(zeros(iteration_number,1));
photons = zeros(N_micro, iteration_number);


for i = 1:iteration_number
    
    for l = 1:N_photoreceptor
        lambdaM = retina_input(l,i)/N_micro;
        photons = poissrnd(lambdaM,1, N_micro)
        for k = 1:N_micro
            X{l,k}(1) = X{l,k}(1) + photons(k);
            while t < t_end
                r1 = rand();
                r2 = rand();

                a_mu = cumsum(a{l,k});

%                 a_s = sum(a{l,k});
                a_s = a_mu(12);
                %dt = 1/(la+a_s)*log(1/r1);
                dt = 1e-4;
        %         if i <= length(T_ph) && t+dt > T_ph(i)
        %             t = T_ph(i);
        %             i = i+1;
        %             X(1) = X(1) + N_ph(1000*t);
        %         else
        %             t = t+dt;
        %         end
                t = t + dt;
                propensity = r2*a_s;
                found = false;
                %for j=randperm(length(c))
                for j=1:12
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
                    %mu_out = [mu_out; j];

                    % update X
                    X{l,k} = X{l,k} + V_state_transition(:,j)';
                    %X_out = [X_out;X];
                end

                % update h
                %C_star = (X(6)/avo)/(V*10^-12);%mM
                C_star = X{l,k}(6);

                C_star_concentration = C_star/Concentration_ratio;
                %C_star_out = [C_star_out;C_star_concentration];
                CaM{l,k} = C_T - C_star;
            
                CaM_concentration = CaM{l,k}/Concentration_ratio;
    %             CaM_out = [CaM_out;CaM];
                h = compute_h(X{l,k}, Ca{l,k}, CaM{l,k});
    %             h_out = [h_out;h];

                % update c
                f_p = calc_f_p(Ca{l,k});
                f_n = calc_f_n(C_star_concentration);
                c{l,k}(1) = gamma_m_star*(1+h_m_star*f_n);
                c{l,k}(7) = gamma_plc_star*(1+h_plc_star*f_n);
                c{l,k}(8) = gamma_d_star*(1+h_d_star*f_n);
                c{l,k}(9) = kappa_t_star*(1+h_t_star_p*f_p)/(k_d_star^2);
                c{l,k}(10) = gamma_t_star*(1+h_t_star_n*f_n);

                % update a
                a{l,k} = h.*c{l,k};
    %             a_out = [a_out; a];

                % update Ca[2+] dynamics
                I_in = I_T_star*X{l,k}(7);
                I_in_output{l,k}(i) = I_in;
                I_Ca = 0.4*I_in;

                f1 = calc_f1(Na_i, Ca_o);
                f2 = calc_f2(V_m, Na_o);

                Ca{l,k} = calc_Ca(C_star_concentration, CaM_concentration, I_Ca, f1, f2);

            end


            t = t_end;
            t_end = t_end + timestep;
            
            
        end
    end
end

I_sum = sum(cell2mat(I_in_output(1,:)), 2);
figure;
plot(I_sum);

%  figure;
%     subplot(4,2,1)
%     plot(X_out(:,1));
%     xlabel('Time');
%     ylabel('M*');
%     
%     subplot(4,2,2)
%     plot(X_out(:,3));
%     xlabel('Time');
%     ylabel('G*');
%     
%     subplot(4,2,3)
%     plot(X_out(:,4));
%     xlabel('Time');
%     ylabel('PLC*');
%     
%     subplot(4,2,4)
%     plot(X_out(:,5));
%     xlabel('Time');
%     ylabel('D*');
%     
%     subplot(4,2,5)
%     plot(X_out(:,7));
%     xlabel('Time');
%     ylabel('T*');
%     
%     subplot(4,2,6)
%     plot(Ca_out);
%     xlabel('Time');
%     ylabel('Ca');
%     
%     subplot(4,2,7)
%     %plot(X_out(:,6));
%     plot(C_star_out);
%     xlabel('Time');
%     ylabel('C*');