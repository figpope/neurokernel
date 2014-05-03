function dx = F_HHN(x,I)

E_K = -85; % potassium reversal potential
E_Cl = -30; % chloride reversal potential
G_s = 1.6; % maximum shaker conductance
G_dr = 3.5; % maximum delayed rectifier conductance
G_Cl = 0.056; % chloride leak conductance
G_K = 0.082; % potassium leak conductance
C = 4; % membrane capacitance

V = x(1);
sa = x(2); % shaker activation
si = x(3); % shaker inactivation
dra = x(4); % delayed rectifier activation
dri = x(5); % delayed rectifier inactivation

% computing voltage gated time constants and steady-state
% activation/inactivation functions
sa_inf = (1./(1+exp((-30-V)/13.5))).^(1/3);
tau_sa = 0.13+3.39*exp(-(-73-V).^2./20^2);
si_inf = 1./(1+exp((-55-V)/-5.5));
tau_si = 113*exp(-(-71-V).^2./29^2);
dra_inf = (1./(1+exp((-5-V)/9))).^(1/2);
tau_dra = 0.5+5.75*exp(-(-25-V).^2./32^2);
dri_inf = 1./(1+exp((-25-V)/-10.5));
tau_dri = 890;

% compute derivatives
dsa = (sa_inf - sa)./tau_sa;
dsi = (si_inf - si)./tau_si;
ddra = (dra_inf - dra)./tau_dra;
ddri = (dri_inf - dri)./tau_dri;
dV = (I - G_K*(V-E_K) - G_Cl * (V-E_Cl) - G_s * sa * si * (V-E_K) - G_dr * dra * dri * (V-E_K) - 0.093*(V-10) )/C;

dx = [dV,dsa,dsi,ddra,ddri];

