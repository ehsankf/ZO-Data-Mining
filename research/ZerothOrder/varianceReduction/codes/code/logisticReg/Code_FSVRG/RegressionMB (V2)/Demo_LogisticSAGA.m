%function Demo_LogisticSAGA(par)
% Demo for Logistic Regression
 
clc; clear;
par = 'ijcnn1';
addpath('Data');
method = '/SAGA/SAGA';
mexinstall;
filename = strcat(par,'.mat'); 
% Data set
%load ('/home/ehsankf/research/ZerothOrder/varianceReduction/codes/code/logisticReg/Code_FSVRG/LogisticRegression Batch/Data/Covtype.mat')
%load ('/home/ehsankf/research/ZerothOrder/varianceReduction/codes/code/logisticReg/Code_FSVRG/LogisticRegression Batch/Data/rcv1_train.binary.mat')
%load adult.mat 
%load ijcnn1.mat
load(filename)
if(par == 'ijcnn1')
 X = x_train;
 y = t_train;
end
if(par == 'ijcnn1')
  X = [ones(size(X,1),1) X];
end
[n, d] = size(X);
X = X';   
% Data set        
%load Covtype.mat  
%[n, d] = size(X);
%X = X';   

% Normalization
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, d, 1);
end
clear sum1

% Parameters
Test = 1; % 1 for 10^(-4) and 2 for 10^(-6). 
Paras = [10^(-4);10^(-6)];  % 10^(-4), 10^(-6)
lambda = Paras(Test); % Regularization parameter
lambda1 = 0; % Regularization parameter
Lmax = (0.25 * max(sum(X.^2,1)) + lambda); % Lipschitz constant

switch par
    case 'mnist'
       eta = 0.4/Lmax;
       outer_loops = 100;
    case  'Covtype' 
       eta = 1 /(2 * Lmax);
       outer_loops = 100;
    otherwise
       eta = 1.00 / (10 * Lmax);
       outer_loops = 10;
end

% SAGA

% outer_loops = 30;
% w = zeros(d, 1);
% eta = 1.00 / (10 * Lmax);
% batch_size = 1;
% tic;
% hist1 = Alg_SAGA(w, full(X), y, lambda, lambda1, eta, outer_loops, batch_size);
% time_SAGA = toc;
% fprintf('Time spent on SAGA: %f seconds \n', time_SAGA);
% x_SAGA = [0:3:3*outer_loops]';
% hist1 = [x_SAGA, hist1];
% clear x_SAGA;
w = zeros(d, 1);
Legend = {};
BatchArray = [n, floor(n/3)];
batch_sizeArray = [1, 2, 5];
nB = size(BatchArray, 2);
mb = size(batch_sizeArray, 2);
for i=1:nB
  for j=1:mb  
    Batch =  BatchArray(1, i);
    batch_size = batch_sizeArray(1, j);
    tic;
    xxoutput= Alg_PSCVR(w, full(X), y, lambda, lambda1, eta, outer_loops, batch_size, Batch);
    hist1(i, j, :) =  xxoutput(1:outer_loops+1);
    zerothhist(i, j, :) =  xxoutput(outer_loops+2:2*outer_loops+2);
    timehist(i, j, :) =  xxoutput(2 * outer_loops+3:3 * outer_loops+3);
    time_PSCVR = toc;
    fprintf('Time spent on PSCVR: %f seconds \n', time_PSCVR);
    Legend((i-1)*mb+j) = strcat('b = ', {' '}, num2str(batch_size), ...
         'B = ', {' '}, num2str(Batch));
  end
end
x_PSCVR = [0:3:3*outer_loops]';
 hist1(end+1,end+1,:) = x_PSCVR;
 clear x_PSCVR

% SVRG
% R. Johnson and T. Zhang:
% "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction", NIPS, 2013.
% outer_loops = 70;
% w = zeros(d, 1);
% eta = 1.00 / (10 * Lmax);
% tic;
% hist2 = Alg_SVRGhist1 = Alg_PSCVR(w, full(X), y, lambda, lambda1, eta, outer_loops, batch_size);
% time_SVRG = toc;
% fprintf('Time spent on SVRG: %f seconds \n', time_SVRG);
% x_SVRG = [0:3:3*outer_loops]';
% hist2 = [x_SVRG, hist2];
% clear x_SVRG; 


% Show results
% Gap vs. Number of effective passes  
%minval = 0.336361591002329; %(lambda=10^(-4)) 
minval =  min(min(abs(hist1(1:end-1, 1:end-1, end))))-1e-10;
%minval = 0.323040236193746; %(lambda=10^(-6)) 
%aa = max(max([hist1(:,2),hist3(:,2)]))-minval; b = 1;
aa = max(squeeze(max(max(hist1(1:end-1, 1:end-1, 1:end)))-minval)); b = 1;
zoQB = min(squeeze(min(min(zerothhist(1:end, 1:end, end)))));
timeQB = max(squeeze(max(max(timehist(1:end, 1:end, end)))));

marker = {'g-.o', 'b:d', 'r--^', 'k-p', 'c-+', 'g-*', 'b-p', 'r-x', 'm-s', 'r-^', 'b-v', 'g->', 'c-<', 'm-h'};
for k=1:3
 h = figure(k);
 set(gcf,'position',[200,100,386,269]) 
 for i=1:nB
  for j=1:mb
    Xres = [squeeze(hist1(end, end, 1:b:end)), squeeze(floor(zerothhist(i, j, 1:b:end)/1000)),...
        squeeze(floor((timehist(i, j, 1:b:end)-timehist(i, j, 1))/1000)), ...
          squeeze(abs(hist1(i, j, 1:b:end)))];
    %Xres = [squeeze(floor(zerothhist(i, j, 1:b:end)/1000)), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    %Xres = [squeeze(floor(timehist(i, j, 1:b:end))), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    name = strcat('results/',par, method, 'B', num2str(i), 'b', num2str(j));
    dlmwrite(name, Xres, 'delimiter', '\t', 'precision', 32);
    Xres = dlmread(name);
    semilogy(Xres(:,k), Xres(:,4)- minval, ...
        char(marker((i-1)*mb+j)), 'DisplayName',char(Legend((i-1)*mb+j)), 'linewidth', 1.6,'markersize',4.5);
%      semilogy(squeeze(hist1(end, end, 1:b:end)), squeeze(abs(hist1(i, j, 1:b:end) - minval(1:b:end)')), ...
%          char(marker((i-1)*m+j)), 'DisplayName',char(Legend((i-1)*m+j)), 'linewidth', 1.6,'markersize',4.5);
    hold on;
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
    
  end
end
hold off
switch k
    case 1
       xlabel('# of epochs');
       axis([0 50, 1E-12,aa])
    case  2 
       xlabel('Queries/1000');
       axis([0 zoQB/1000, 1E-12,aa]);
    case  3 
       xlabel('CPU time (seconds)');
       axis([0 timeQB/1000, 1E-12,aa]);

end

ylabel('Objective minus best');
% xlabel('Number of effective passes');
% ylabel('Objective minus best');
% axis([0 outer_loops, 1E-12,aa])
%legend('SAGA','SVRG','Katyusha','FSVRG');

legend(Legend)
figname = strcat('results/',par, method, 'B', num2str(i), 'b',...
    num2str(j));
saveas(h, strcat(figname,'fig', num2str(k),'.pdf'))
end

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



