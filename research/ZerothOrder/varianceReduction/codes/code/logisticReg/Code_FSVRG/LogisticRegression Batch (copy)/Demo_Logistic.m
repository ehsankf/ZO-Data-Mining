% Demo for Logistic Regression
 
clc; clear;
mexinstall;

% Data set
load ('/home/ehsankf/research/ZerothOrder/varianceReduction/codes/code/logisticReg/Code_FSVRG/Data/adult.mat')
[n, d] = size(X);
X = X';   

% Parameters
Test = 1; % 1 for 10^(-4) and 2 for 10^(-6). 
Paras = [10^(-4);10^(-6)];  % 10^(-4), 10^(-6)
lambda = Paras(Test); % Regularization parameter
lambda1 = 0; % Regularization parameter
Lmax = (0.25 * max(sum(X.^2,1)) + lambda); % Lipschitz constant



% SAGA

outer_loops = 20;
w = zeros(d, 1);
eta = 1.00 / (10 * Lmax);
batch_size = 20;
tic;
hist1 = Alg_SAGA(w, full(X), y, lambda, lambda1, eta, outer_loops, batch_size);
time_SAGA = toc;
fprintf('Time spent on SAGA: %f seconds \n', time_SAGA);
x_SAGA = [0:3:3*outer_loops]';
hist1 = [x_SAGA, hist1];
clear x_SAGA; 

% SVRG
% R. Johnson and T. Zhang:
% "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction", NIPS, 2013.
% outer_loops = 70;
% w = zeros(d, 1);
% eta = 1.00 / (10 * Lmax * d);
% tic;
% hist2 = Alg_SVRG(w, full(X), y, lambda, eta, outer_loops);
% time_SVRG = toc;
% fprintf('Time spent on SVRG: %f seconds \n', time_SVRG);
% x_SVRG = [0:3:3*outer_loops]';
% hist2 = [x_SVRG, hist2];
% clear x_SVRG; 




% SVRG++
% Z. Allen-Zhu and Y. Yuan:
% "Improved SVRG for Non-Strongly-Convex or Sum-of-Non-Convex Objectives", ICML, 2016.
% % outer_loops = 10;
% % eta = 1 / (7 * Lmax);
% % tic;
% % hist2 = Alg_SVRGAA(w, full(X), y, lambda, eta, outer_loops);
% % time_SVRGAA = toc;
% % fprintf('Time spent on SVRG++: %f seconds \n', time_SVRGAA);
% % a  = 0;
% % for i=1:outer_loops
% %     a = [a;round(n/4)*(2^(i-1))/n+1];
% % end
% % x_SVRGAA = [];
% % for i = 1:outer_loops+1
% %     x_SVRGAA(i) = sum(a(1:i));
% % end
% % clear a 
% % hist2 = [x_SVRGAA', hist2];
% % clear x_SVRGAA;

% Katyusha
% Z. Allen-Zhu: "Katyusha: The First Direct Acceleration of Stochastic Gradient Methods",
% Journal of Machine Learning Research, 2017.
% outer_loops = 70;
% tic;
% hist3 = Alg_Katyusha(w, full(X), y, lambda, Lmax, outer_loops);
% time_Katyusha = toc;
% fprintf('Time spent on Katyusha: %f seconds \n', time_Katyusha);
% x_Katyusha = [0:3:3*outer_loops]';
% hist3 = [x_Katyusha, hist3];
% clear x_Katyusha; 

% Our FSVRG method 
outer_loops = 13;
% eta = 1.5/Lmax;
% theta = [0.99,0.80]; theta = theta(Test);
% tic;
% hist4 = Alg_FSVRG(w, full(X), y, lambda, eta, theta, outer_loops);
% time_FSVRG = toc;
% fprintf('Time spent on FSVRG: %f seconds \n', time_FSVRG);
% a  = [0;0.25];
% for i = 2:outer_loops
%     a = [a;round(n/4)*(1.65^(i-2))/n+1];
% end
% x_FSVRG = [];
% for i = 1:outer_loops+1
%     x_FSVRG(i) = sum(a(1:i));
% end
% hist4 = [x_FSVRG', hist4];
% clear x_FSVRG;
%     

% Show results
% Gap vs. Number of effective passes  
minval = 0.336361591002329; %(lambda=10^(-4)) 
%minval = 0.323040236193746; %(lambda=10^(-6)) 
%aa = max(max([hist1(:,2),hist3(:,2)]))-minval; b = 1;
aa = max(max([hist1(:,2),hist1(:,2)]))-minval; b = 1;
h = figure(11)
set(gcf,'position',[200,100,386,269]) 
semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'g-.o','linewidth',1.6,'markersize',4.5);
% hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
% hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
% hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
hold off
xlabel('Number of effective passes');
ylabel('Objective minus best');
axis([0 25, 1E-12,aa])
legend('SAGA','SVRG','Katyusha','FSVRG');
saveas(h,'fig11.pdf')

% Gap vs. Time (seconds)
%a1 = time_SVRG; hist11(:,1) = [0:a1/(size(hist1,1)-1):a1]';
%a2 = time_SVRGAA; hist12(:,1) = a2*hist2(:,1)/max(hist2(:,1));
%a3 = time_Katyusha; hist13(:,1) = [0:a3/(size(hist3,1)-1):a3]';
%a4 = time_FSVRG; hist14(:,1) = a4*hist4(:,1)/max(hist4(:,1));
%h = figure (12)
%set(gcf,'position',[200,100,386,269])
%semilogy(hist11(1:b:end,1), abs(hist1(1:b:end,2) - minval),'g-.o','linewidth',1.6,'markersize',4.5); 
%hold on,semilogy(hist12(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
%hold on,semilogy(hist13(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
%hold on,semilogy(hist14(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
%hold off
%xlabel('Running time (sec)');
%ylabel('Objective minus best');
%axis([0 1, 1E-12,aa])
%legend('SVRG','SVRG++','Katyusha','FSVRG');
%saveas(h,'fig12.pdf')
%clear a1 a2 a3 a4;



