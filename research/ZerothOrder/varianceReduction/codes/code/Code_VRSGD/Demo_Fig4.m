%%% Demo for comparison of Options I, II and III in Table 1
%
% If you have any questions about the code, 
% feel free to contact Fanhua Shang: fhshang@cse.cuhk.edu.hk 
%
%

clc; clear;
addpath('/home/ehsankf/research/ZerothOrder/varianceReduction/codes/code/Code_VRSGD/Algorithms');
addpath('Datasets');
Mexinstall_fig4;

% Data set        
load Covtype.mat  
[n, d] = size(X);
X = X';   

% Normalization
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, d, 1);
end
clear sum1

% Parameters
lambda = 10^(-5); % Regularization parameter
Lmax = max(sum(X.^2,1)) + lambda; % Lipschitz constant
outer_loops = 3000; % Number of epoch
eta = 1 /(2 * Lmax); % Learning rate
w = zeros(d, 1); % To initiate the variable, w

% OptionI
tic
hist1 = Alg_OptionI(w, full(X), y, lambda, eta, outer_loops);
time1 = toc;
fprintf('Time spent on Option-I: %f seconds \n', time1);
x_SVRG = [0:3:3*outer_loops]';
hist1  = [x_SVRG, hist1];

% OptionII
tic
hist2 = Alg_OptionII(w, full(X), y, lambda, eta, outer_loops);
time2 = toc;
fprintf('Time spent on Option-II: %f seconds \n', time2);
hist2  = [x_SVRG, hist2];
clear x_SVRG;

% OptionIII
eta = 4/(3 * Lmax);  
tic
hist3 = Alg_OptionIII(w, full(X), y, lambda, eta, outer_loops);
time3 = toc;
fprintf('Time spent on Option-III: %f seconds \n', time3);
x_VRSGD = [0:3:3*outer_loops]';
hist3   = [x_VRSGD, hist3];
clear x_VRSGD;


%%% Results
% Objective gap vs number of effective passes 
minval = 0.475329389248235; maxval = max([hist1(:,2)]);
aa = maxval - minval; b = 1;
figure(4)
set(gcf,'position',[200,100,386,269])  
semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'g-.o','linewidth',1.6);
hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b--^','linewidth',1.6);
hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r-d','linewidth',1.2); 
hold off
xlabel('Gradient evaluations / \it{n}');
ylabel('\it{F} \rm{(}\it{x^{s}}\rm{ ) -} \it{F} \rm{(}\it{x}^{*}\rm{ )}');
axis([0 30 1E-12 aa])
legend('Option I','Option II','Option III');
clear aa maxval minval; 




