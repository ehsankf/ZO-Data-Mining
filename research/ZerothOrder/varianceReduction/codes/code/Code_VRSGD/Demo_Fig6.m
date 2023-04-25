%%% Demo for Comparison of SVRG++, VR-SGD and VR-SGD++
%
% If you have any questions about the code, 
% feel free to contact Fanhua Shang: fhshang@cse.cuhk.edu.hk 
% 
%

clc; clear;
addpath('Algorithms');
addpath('Datasets');
%Mexinstall_fig6;

% Data set
load Adult.mat        
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
lambda = 10^(-5); %Regularization parameter 
Lmax = (0.25 * max(sum(X.^2,1)) + lambda); % Lipschitz constant
outer_loops = 50; % Number of epoch
eta = 3.5/(10 * Lmax); % Learning rate  
w = zeros(d, 1); % To initiate the variable, w


% VR-SGD
tic
hist1 = Alg_VRSGD(w, full(X), y, lambda, eta, outer_loops);
time1 = toc;
fprintf('Time spent on VRSGD: %f seconds \n', time1);
x_VRSGD = [0:3:3*outer_loops]';
hist1   = [x_VRSGD, hist1];
clear x_VRSGD;

% VR-SGD++ 
% VR-SGD with a growing epoch size strategy in early iterations
tic
hist2 = Alg_VRSGDAA(w, full(X), y, lambda, eta, outer_loops);
time2 = toc;
fprintf('Time spent on VRSGD++: %f seconds \n', time2);
a  = 0;
for i = 1:outer_loops
    if i < 5
        a = [a;round(n/4)*(1.75^(i-1))/n+1];
    else
       a = [a; 3]; 
    end
end
x_VRSGD = [];
for i = 1:outer_loops+1
    x_VRSGD(i) = sum(a(1:i));
end
clear a 
hist2 = [x_VRSGD', hist2];
clear x_VRSGD;

% SVRG++
% Z. Allen-Zhu and Y. Yuan:
% Improved SVRG for Non-Strongly-Convex or Sum-of-Non-Convex Objectives, ICML, 2016.
outer_loops = 10;
tic;
hist3 = Alg_SVRGAA(w, full(X), y, lambda, eta, outer_loops);
time3 = toc;
fprintf('Time spent on SVRG++: %f seconds \n', time3);
a  = 0;
for i=1:outer_loops
    a = [a;round(n/4)*(2^(i-1))/n+1];
end
x_SVRGAA = [];
for i = 1:outer_loops+1
    x_SVRGAA(i) = sum(a(1:i));
end
clear a 
hist3 = [x_SVRGAA', hist3];
clear x_SVRGAA;


% Results
% Objective gap vs number of effective passes    
minval = 0.326527714774212; maxval = max([hist1(:,2)]);
aa = maxval - minval; b = 1;
figure(6)
set(gcf,'position',[200,100,386,269])  
semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'b--o','linewidth',1.6,'markersize',5);
hold on,semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'g-d','linewidth',1.2,'markersize',5); 
hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'r-.s','linewidth',1.6,'markersize',5); 
hold off
xlabel('Gradient evaluations / \it{n}');
ylabel('\it{F} \rm{(}\it{x^{s}}\rm{ ) -} \it{F} \rm{(}\it{x}^{*}\rm{ )}');
axis([0 50 1E-12 aa])
legend('SVRG++','VR-SGD','VR-SGD++');
clear aa maxval minval; 


