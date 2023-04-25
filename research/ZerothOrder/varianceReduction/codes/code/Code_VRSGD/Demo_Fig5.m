%%% Demo for comparison of commoen stochastic update rules and proximal update rules 
%
% If you have any questions about the code, 
% feel free to contact Fanhua Shang: fhshang@cse.cuhk.edu.hk 
%
%
clc; clear;
addpath('Algorithms');
addpath('Datasets');
%Mexinstall_fig5;

% Data set
load Adult.mat              
[n, d] = size(X);
X = X';   

% Normalization
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, d, 1);
end
clear sum1

% Parameters
lambda = 10^(-4); %{10^(-3),10^(-4),10^(-5),10^(-6),10^(-7)}
Lmax = max(sum(X.^2,1)) + lambda;
outer_loops = 30; % Number of epoch
eta = 1/(5 * Lmax); % Learning rate 
w = zeros(d, 1); % To initiate the variable, w
    

% VR-SGDI
% VR-SGD with common update rules
tic
hist1 = Alg_VRSGDI(w, full(X), y, lambda, eta, outer_loops);
time1 = toc;
fprintf('Time spent on VRSGD-I: %f seconds \n', time1);
x_VRSGD = [0:3:3*outer_loops]';
hist1 = [x_VRSGD, hist1];

% VR-SGDII
% VR-SGD with proximal update rules
tic
hist2 = Alg_VRSGDII(w, full(X), y, lambda, eta, outer_loops);
time2 = toc;
fprintf('Time spent on VRSGD-II: %f seconds \n', time2);
hist2 = [x_VRSGD, hist2];
clear x_VRSGD;


% SVRG-I
% SVRG with common update rules
tic
hist3 = Alg_SVRGI(w, full(X), y, lambda, eta, outer_loops);
time3 = toc;
fprintf('Time spent on SVRG-I: %f seconds \n', time3);
x_SVRG = [0:3:3*outer_loops]';
hist3  = [x_SVRG, hist3];

% SVRG-II
% SVRG with proximal update rules
tic
hist4 = Alg_SVRGII(w, full(X), y, lambda, eta, outer_loops);
time4 = toc;
fprintf('Time spent on SVRG-II: %f seconds \n', time4);
hist4  = [x_SVRG, hist4];
clear x_SVRG;


% Katyusha_I
% Katyusha with common update rules
tic;
hist5 = Alg_KatyushaI(w, full(X), y, lambda, Lmax, outer_loops);
time5 = toc;
fprintf('Time spent on Katyusha-I: %f seconds \n', time5);
x_Katyusha = [0:3:3*outer_loops]';
hist5 = [x_Katyusha, hist5];

% Katyusha_II
% Katyusha with proximal update rules
tic;
hist6 = Alg_KatyushaII(w, full(X), y, lambda, Lmax, outer_loops);
time6 = toc;
fprintf('Time spent on Katyusha-II: %f seconds \n', time6);
hist6 = [x_Katyusha, hist6];
clear x_Katyusha;


%%% Results
% Objective gap vs number of effective passes   
minval = 0.225533846816392; maxval = max([hist1(:,2)]);
aa = maxval - minval; b = 1;
figure(51)
set(gcf,'position',[200,100,386,269])  
semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'g-.o','linewidth',1.6,'markersize',5);
hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'g--^','linewidth',1.6,'markersize',5);
hold on,semilogy(hist5(1:b:end,1), abs(hist5(1:b:end,2) - minval),'b-.o','linewidth',1.6,'markersize',5);
hold on,semilogy(hist6(1:b:end,1), abs(hist6(1:b:end,2) - minval),'b--^','linewidth',1.6,'markersize',5);
b = 1;
hold on,semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'r-.o','linewidth',1.6,'markersize',5); 
hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',5); 
hold off
xlabel('Gradient evaluations / \it{n}');
ylabel('\it{F} \rm{(}\it{x^{s}}\rm{ ) -} \it{F} \rm{(}\it{x}^{*}\rm{ )}');
axis([0 40 1E-12 aa])
legend('SVRG-I','SVRG-II','Katyusha-I','Katyusha-II','VR-SGD-I','VR-SGD-II');

% Objective gap vs time (seconds)
hist11(:,1) = [0:time1/(size(hist1,1)-1):time1]';
hist12(:,1) = [0:time2/(size(hist2,1)-1):time2]';
hist13(:,1) = [0:time3/(size(hist3,1)-1):time3]';
hist14(:,1) = [0:time4/(size(hist4,1)-1):time4]';
hist15(:,1) = [0:time5/(size(hist5,1)-1):time5]';
hist16(:,1) = [0:time6/(size(hist6,1)-1):time6]';
b = 1;
figure (52)
set(gcf,'position',[200,100,386,269])
semilogy(hist13(1:b:end,1), abs(hist3(1:b:end,2) - minval),'g-.o','linewidth',1.6,'markersize',5);
hold on,semilogy(hist14(1:b:end,1), abs(hist4(1:b:end,2) - minval),'g--^','linewidth',1.6,'markersize',5);
hold on,semilogy(hist15(1:b:end,1), abs(hist5(1:b:end,2) - minval),'b-.o','linewidth',1.6,'markersize',5);
hold on,semilogy(hist16(1:b:end,1), abs(hist6(1:b:end,2) - minval),'b--^','linewidth',1.6,'markersize',5);
hold on,semilogy(hist11(1:b:end,1), abs(hist1(1:b:end,2) - minval),'r-.o','linewidth',1.6,'markersize',5); 
hold on,semilogy(hist12(1:b:end,1), abs(hist2(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',5); 
hold off
xlabel('Running time (sec)');
ylabel('\it{F} \rm{(}\it{x^{s}}\rm{ ) -} \it{F} \rm{(}\it{x}^{*}\rm{ )}');
axis([0 1 1E-12 aa])
legend('SVRG-I','SVRG-II','Katyusha-I','Katyusha-II','VR-SGD-I','VR-SGD-II');
clear aa maxval minval; 


