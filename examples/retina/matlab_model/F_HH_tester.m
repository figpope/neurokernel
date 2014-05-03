dt = 1e-5;

X = [-70,0.3664,0.8969,0.027,0.4092];
% X = [0,0,0,0,0];
X_dv = zeros(1,length(X));
time = 0.5
step = floor(time/dt);
% I = zeros(1,step);
% I(step/4:step*3/4) = 10;
t = 0:dt:time;
I = input_gen(2*pi*50,5,5,t);
% I_sum_4 = cell2mat(arrayfun(@(x) repmat(x, 1, 100), I_sum, 'UniformOutput', false)
% )
% I_sum_2 = arrayfun(@(x) repmat(x, 1, 10), I_sum)
% I_sum_2 = repmat(I_sum, 10,1);
% for i = 1:10
%     I_sum_2(i:10:end) = I_sum(1:end);   
% end



% I = I_sum5/(1.57*10^(-5))/1e6+5;
% I = zeros(size(t));
% I = I_sum;
I = repmat(I_sum, 1, 10)
X_out = X;
for i=1:step
    X = X + X_dv*dt*1000;
    X_dv = F_HHN(X,I(i));
    X_out = [X_out;X];
end
figure;
plot(X_out(:,1));

