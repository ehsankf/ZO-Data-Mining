%%% Demo for Figure 3
%
% If you have any questions about the code, 
% feel free to contact Fanhua Shang: fhshang@cse.cuhk.edu.hk 
% 
%

clc; clear;
addpath('Algorithms');
addpath('Datasets');
%Mexinstall_fig3; 

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
lambda = 1*10^(-5); % Regularization parameter
Lmax = (0.25 * max(sum(X.^2,1)) + lambda); % Lipschitz constant
outer_loops = 100; % Number of epoch
w = zeros(d, 1); % To initiate the variable, w

% VR-SGD
eta = 2/(5*Lmax);  
tic
hist1 = Alg_VRSGD(w, full(X), y, lambda, eta, outer_loops);
time1 = toc;
fprintf('Time spent on VRSGDI: %f seconds \n', time1);
x_VRSGD = [0:3:3*outer_loops]';
hist1 = [x_VRSGD, hist1];
clear x_VRSGD;

% SVRG
% R. Johnson and T. Zhang:
% "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction", NIPS, 2013.
eta = 1/(10*Lmax);  
tic
hist2 = Alg_SVRG(w, full(X), y, lambda, eta, outer_loops);
time2 = toc;
fprintf('Time spent on SVRG: %f seconds \n', time2);
x_SVRG = [0:3:3*outer_loops]';
hist2 = [x_SVRG, hist2];
clear x_SVRG;

% Katyusha
% Zeyuan Allen-Zhu: "Katyusha: The First Direct Acceleration of Stochastic
% Gradient Methods", JMLR, 2017.
tic;
hist3 = Alg_Katyusha(w, full(X), y, lambda, Lmax, outer_loops);
time3 = toc;
fprintf('Time spent on Katyusha: %f seconds \n', time3);
x_Katyusha = [0:3:3*outer_loops]';
hist3 = [x_Katyusha, hist3];
clear x_Katyusha;

% AGD
outer_loops = 500;
beta = (sqrt(Lmax)-sqrt(lambda))/(sqrt(Lmax)+sqrt(lambda));
tic;
hist4 = Alg_AGD(w, full(X), y, lambda, 0.50/Lmax, beta, outer_loops);
time4 = toc;
fprintf('Time spent on AGD: %f seconds \n', time4);
x_AGD = [0:1:1*outer_loops]';
hist4 = [x_AGD, hist4];
clear x_AGD;

% SGD
outer_loops = 200;
eta = 0.25/Lmax;
tic;
hist5 = Alg_SGD(w, full(X), y, lambda, eta, outer_loops);
time5 = toc;
fprintf('Time spent on SGD: %f seconds \n', time5);
x_SGD = [0:2:2*outer_loops]';
hist5 = [x_SGD, hist5];
clear x_SGD;


% Results
% Objective gap vs number of effective passes  
minval = 0.326527714774211; maxval = max([hist1(:,2)]);
aa = maxval - minval; b = 2;
figure(31)
set(gcf,'position',[200,100,386,269])
semilogy(hist5(1:b:end,1), abs(hist5(1:b:end,2) - minval),'m-s','linewidth',1.6,'markersize',5);
b = 8;
hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'b--^','linewidth',1.6,'markersize',5);
b = 1;
hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g-.o','linewidth',1.6,'markersize',5);
hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r:+','linewidth',2.6,'markersize',5);
hold on,semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'k-d','linewidth',1.2,'markersize',5); 
hold off
xlabel('Gradient evaluations / \it{n}');
ylabel('\it{F} \rm{(}\it{x^{s}}\rm{ ) -} \it{F} \rm{(}\it{x}^{*}\rm{ )}');
axis([0 102 1E-12 aa])
legend('SGD','AGD','SVRG','Katyusha','VR-SGD','location','East');

% Objective gap vs running time (seconds)
hist11(:,1) = [0:time1/(size(hist1,1)-1):time1]';
hist12(:,1) = [0:time2/(size(hist2,1)-1):time2]';
hist13(:,1) = [0:time3/(size(hist3,1)-1):time3]';
hist14(:,1) = [0:time4/(size(hist4,1)-1):time4]';
hist15(:,1) = [0:time5/(size(hist5,1)-1):time5]';
b = 4;
figure (32)
set(gcf,'position',[200,100,386,269])
semilogy(hist15(1:b:end,1), abs(hist5(1:b:end,2) - minval),'m-s','linewidth',1.6,'markersize',5);
b = 16;
hold on,semilogy(hist14(1:b:end,1), abs(hist4(1:b:end,2) - minval),'b--^','linewidth',1.6,'markersize',5);
b = 2;
hold on,semilogy(hist12(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g-.o','linewidth',1.6,'markersize',5);
b = 1;
hold on,semilogy(hist13(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r:+','linewidth',2.6,'markersize',5);
hold on,semilogy(hist11(1:b:end,1), abs(hist1(1:b:end,2) - minval),'k-d','linewidth',1.2,'markersize',5); 
hold off
xlabel('Running time (sec)');
ylabel('\it{F} \rm{(}\it{x^{s}}\rm{ ) -} \it{F} \rm{(}\it{x}^{*}\rm{ )}');
axis([0 2 1E-12 aa])
legend('SGD','AGD','SVRG','Katyusha','VR-SGD','location','East');
clear aa maxval minval; 


