%%% Demo for robustness of choosing learning rates
%
% If you have any questions about the code, 
% feel free to contact Fanhua Shang: fhshang@cse.cuhk.edu.hk 
% 

clc; clear;

%addpath('Algorithms');
addpath('../Data');
mexinstall; % 

% Data set
load Covtype.mat  
X = [ones(size(X,1),1) X];
[n, d] = size(X);
X = X'; 

% Normalization
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, d, 1);
end
clear sum1

% Parameters
lambda = 10^(-4); % Regularization parameter
Lmax = (0.25 * max(sum(X.^2,1)) + lambda); % Lipschitz constant
outer_loops = 30; % Number of epoch
w = zeros(d, 1); % To initiate the variable, w

eta = 1/(5*Lmax); % Please choose a learning rate from {1/(5L),6/(5L)}
% VR-SGD with a fixed learnig rate, eta
tic
hist1 = Alg_SVRG(w, full(X), y, lambda, eta, outer_loops);
time1 = toc;
fprintf('Time spent on VRSGD: %f seconds \n', time1);
x_VRSGD = [0:3:3*outer_loops]';
hist1 = [x_VRSGD, hist1];
w = zeros(d, 1);
% SVRG
% R. Johnson and T. Zhang:
% % "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction", NIPS, 2013.
tic
hist2 = Alg_SVRG(w, full(X), y, lambda, eta, outer_loops);
time2 = toc;
fprintf('Time spent on SVRG: %f seconds \n', time2);
x_SVRG = [0:3:3*outer_loops]';
hist2 = [x_SVRG, hist2];
% 
% 
eta = 6/(5*Lmax); % Please choose a learning rate from {1/(5L),6/(5L)}
% VR-SGD with a fixed learnig rate, eta
tic
hist3 = Alg_SVRG(w, full(X), y, lambda, eta, outer_loops);
time3 = toc;
fprintf('Time spent on VRSGD: %f seconds \n', time3);
x_VRSGD = [0:3:3*outer_loops]';
hist3 = [x_VRSGD, hist3];
% 
% % SVRG
% % R. Johnson and T. Zhang:
% % "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction", NIPS, 2013.
% tic
% hist4 = Alg_SVRG(w, full(X), y, lambda, eta, outer_loops);
% time4 = toc;
% fprintf('Time spent on SVRG: %f seconds \n', time4);
% x_SVRG = [0:3:3*outer_loops]';
% hist4 = [x_SVRG, hist4];


%%% Results
% Objective gap vs number of effective passes 
minval = 0.677043159445869; maxval = max([hist1(:,2)]);
aa = maxval - minval; b = 1;
figure(1)
set(gcf,'position',[200,100,416,309])   
semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b-.d','linewidth',1.6); 
% hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'b--o','linewidth',1.6);
hold on,semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'r-.d','linewidth',1.6);
hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--o','linewidth',1.6); 
hold off
xlabel('Number of effective passes');
ylabel('Objective minus best');
axis([0 35 1E-12 aa])
legend('SVRG, \eta=0.2/L','SVRG, \eta=1.2/L','VR-SGD, \eta=0.2/L','VR-SGD, \eta=1.2/L');
clear aa maxval minval;