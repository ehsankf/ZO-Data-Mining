% Demo for comparison of Algorithms 2 and 3 
%
% If you have any questions about the code, 
% feel free to contact Fanhua Shang: fhshang@cse.cuhk.edu.hk 
% 
 
clc; clear;

addpath('Algorithms');
addpath('Datasets');
% Mexinstall_fig2;

% Data set 
load mnist.mat
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
lambda = 0; % Regularization parameter
Lmax = (0.25 * max(sum(X.^2,1)) + lambda); % Lipschitz constant
outer_loops = 70; % Number of epoch
w = zeros(d, 1); % To initiate the variable, w
eta = 0.4/Lmax; % Learning rate


% Algorithm 3 with Option I
tic;
hist1 = Algorithm3I(w, full(X), y, lambda, eta, outer_loops);
time1 = toc;
fprintf('Time spent on Algorithm3I: %f seconds \n', time1);
x_Algorithm3 = [0:3:3*outer_loops]';
hist1 = [x_Algorithm3, hist1];


% Algorithm 3 with Option II
tic
hist2 = Algorithm3II(w, full(X), y, lambda, eta, outer_loops);
time2 = toc;
fprintf('Time spent on Algorithm3II: %f seconds \n', time2);
hist2 = [x_Algorithm3, hist2];
clear x_Algorithm3;


% VRSGD
tic;
hist3 = Alg_VRSGD(w, full(X), y, lambda, eta, outer_loops);
time3 = toc;
fprintf('Time spent on VRSGD: %f seconds \n', time3);
x_VRSGD = [0:3:3*outer_loops]';
hist3 = [x_VRSGD, hist3];
clear x_VRSGD;


% Katyusha
% Zeyuan Allen-Zhu: "Katyusha: The First Direct Acceleration of Stochastic
% Gradient Methods", JMLR, 2017.
tic;
hist4 = Alg_Katyusha2(w, full(X), y, lambda, Lmax, outer_loops);
time4 = toc;
fprintf('Time spent on Katyusha: %f seconds \n', time4);
x_Katyusha = [0:3:3*outer_loops]';
hist4 = [x_Katyusha, hist4];
clear x_Katyusha;


% Show results
% Gap vs. Number of effective passes 
minval = 0.018149303058103; maxval = max([hist1(:,2)]);
aa = maxval - minval; b = 3;
figure(21)
set(gcf,'position',[200,100,386,269]) 
semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'b-.^','linewidth',1.6,'markersize',4.5);
hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g--o','linewidth',1.6,'markersize',4.5); 
hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r-d','linewidth',1.2,'markersize',4.5);
hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k:s','linewidth',2,'markersize',4.5);
hold off
xlabel('Number of effective passes');
ylabel('Objective minus best');
axis([0 205, 1.6E-4,aa])
legend('Algorithm 3 (Option I)','Algorithm 3 (Option II)','VR-SGD','Katyusha');


% Objective gap vs running time (seconds)
hist11(:,1) = [0:time1/(size(hist1,1)-1):time1]';
hist12(:,1) = [0:time2/(size(hist2,1)-1):time2]';
hist13(:,1) = [0:time3/(size(hist3,1)-1):time3]';
hist14(:,1) = [0:time4/(size(hist4,1)-1):time4]';
figure (22)
set(gcf,'position',[200,100,386,269])
semilogy(hist11(1:b:end,1), abs(hist1(1:b:end,2) - minval),'b-.^','linewidth',1.6,'markersize',4.5);
hold on,semilogy(hist12(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g--o','linewidth',1.6,'markersize',4.5);
hold on,semilogy(hist13(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r-d','linewidth',1.2,'markersize',4.5); 
hold on,semilogy(hist14(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k:s','linewidth',2,'markersize',4.5); 
hold off
xlabel('Running time (sec)');
ylabel('Objective minus best');
axis([0 40 1.6E-4 aa])
legend('Algorithm 3 (Option I)','Algorithm 3 (Option II)','VR-SGD','Katyusha');
clear aa maxval minval; 

