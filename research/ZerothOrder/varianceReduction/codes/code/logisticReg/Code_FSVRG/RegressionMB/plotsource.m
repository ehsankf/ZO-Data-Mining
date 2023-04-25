%function Demo_LogisticSAGA(par)
% Demo for Logistic Regression
 
clc; clear;
close all;

width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 4;

par = 'adult';
% par = 'w8a';
% par = 'mnist';
%  par = 'ijcnn1';
addpath('Data');
addpath('/home/ehsankf/research/code/export_fig-master')
% mexinstall;
filename = strcat(par,'.mat'); 
% Data set
%load ('/home/ehsankf/research/ZerothOrder/varianceReduction/codes/code/logisticReg/Code_FSVRG/LogisticRegression Batch/Data/Covtype.mat')
%load ('/home/ehsankf/research/ZerothOrder/varianceReduction/codes/code/logisticReg/Code_FSVRG/LogisticRegression Batch/Data/rcv1_train.binary.mat')
%load adult.mat 
%load ijcnn1.mat
load(filename)
if(strcmp(par,'ijcnn1'))
 X = x_train;
 y = t_train;
end
if(strcmp(par,'w8a'))
 X = data(:,1:end-1);
 y = data(:,end);
end
if(strcmp(par,'mnist'))
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
BatchArray = [n, floor(n/5)];
% batch_sizeArray = [5, 10, 50];
% batch_sizeArray = [5];
% batch_sizeArray = [10];
batch_sizeArray = [50];
innerloop = 5;
nB = size(BatchArray, 2);
mb = size(batch_sizeArray, 2);
% for i=1:nB
%   for j=1:mb  
%     Batch =  BatchArray(1, i);
%     batch_size = batch_sizeArray(1, j);
%     tic;
%     xxoutput= Alg_PSCVR(w, full(X), y, lambda, lambda1, eta, outer_loops, batch_size, Batch, innerloop);
%     hist1(i, j, :) =  xxoutput(1:outer_loops+1);
%     zerothhist(i, j, :) =  xxoutput(outer_loops+2:2*outer_loops+2);
%     timehist(i, j, :) =  xxoutput(2 * outer_loops+3:3 * outer_loops+3);
%     time_PSCVR = toc;
%     fprintf('Time spent on PSCVR: %f seconds \n', time_PSCVR);
%     Legend((i-1)*mb+j) = strcat('b = ', {' '}, num2str(batch_size), ...
%          'B = ', {' '}, num2str(Batch));
%   end
% end
% x_PSCVR = [0:3:3*outer_loops]';
%  hist1(end+1,end+1,:) = x_PSCVR;
%  clear x_PSCVR

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

%minval = 0.323040236193746; %(lambda=10^(-6)) 
%aa = max(max([hist1(:,2),hist3(:,2)]))-minval; b = 1;
minval = 1;
aa = 0;
zoQB = 1e10;
timeQB = 0;

marker = {'r-.*', 'g-.o', 'b-.>', 'k-p', 'c-+', 'g-*', 'b-p', 'r-x', 'm-s', 'r-^', 'b-v', 'g->', 'c-<', 'm-h'};
method = '/PSCVR/PSCVR';
scheme = 'ZO-PSVRG+';
 for i=1:nB
  for j=1:mb
      if (i ==2)
          scheme = 'ZO-PSVRG';
      end
%     Xres = [squeeze(hist1(end, end, 1:b:end)), squeeze(floor(zerothhist(i, j, 1:b:end)/1000)),...
%         squeeze(floor((timehist(i, j, 1:b:end)-timehist(i, j, 1))/1000)), ...
%           squeeze(abs(hist1(i, j, 1:b:end)))];
    %Xres = [squeeze(floor(zerothhist(i, j, 1:b:end)/1000)), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    %Xres = [squeeze(floor(timehist(i, j, 1:b:end))), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    Batch =  BatchArray(1, i);
    batch_size = batch_sizeArray(1, j);
    Legend((i-1)*mb+j) = strcat(scheme, ' b = ', {' '}, num2str(batch_size));
    name = strcat('results/',par, method, 'B', num2str(i), 'b', num2str(j));
%     dlmwrite(name, Xres, 'delimiter', '\t', 'precision', 32);
    Xresinstant = dlmread(name, '\t');
    Xres((i-1)*mb+j,:, :) = Xresinstant;
    minval =  min(min(Xres((i-1)*mb+j, 1:end, 4)), minval);
    aa = max(squeeze(max(max(Xres((i-1)*mb+j, 1:end, 4)))-minval)); b = 1;
    zoQB = min(squeeze(min(min(Xres((i-1)*mb+j, end, 2)))), zoQB);
    timeQB = max(squeeze(max(max(Xres((i-1)*mb+j, 1:end, 3)))), timeQB);
    %Xres = dlmread(name);
    
%      semilogy(squeeze(hist1(end, end, 1:b:end)), squeeze(abs(hist1(i, j, 1:b:end) - minval(1:b:end)')), ...
%          char(marker((i-1)*m+j)), 'DisplayName',char(Legend((i-1)*m+j)), 'linewidth', 1.6,'markersize',4.5);
    
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
    
  end
 end
 method = '/PSCVRRGE/PSCVRRGE';
scheme = 'ZO-PSVRG+ (RandSGE)';
 for i=1:nB
  for j=1:mb
      if (i ==2)
          scheme = 'ZO-PSVRG (RandSGE)';
      end
%     Xres = [squeeze(hist1(end, end, 1:b:end)), squeeze(floor(zerothhist(i, j, 1:b:end)/1000)),...
%         squeeze(floor((timehist(i, j, 1:b:end)-timehist(i, j, 1))/1000)), ...
%           squeeze(abs(hist1(i, j, 1:b:end)))];
    %Xres = [squeeze(floor(zerothhist(i, j, 1:b:end)/1000)), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    %Xres = [squeeze(floor(timehist(i, j, 1:b:end))), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    Batch =  BatchArray(1, i);
    batch_size = batch_sizeArray(1, j);
    Legend((i-1)*mb+j) = strcat(scheme, ' b = ', {' '}, num2str(batch_size));
    name = strcat('results/',par, method, 'B', num2str(i), 'b', num2str(j));
%     dlmwrite(name, Xres, 'delimiter', '\t', 'precision', 32);
    Xresinstant = dlmread(name, '\t');
    Xres((i-1)*mb+j,:, :) = Xresinstant;
    minval =  min(min(Xres((i-1)*mb+j, 1:end, 4)), minval);
    aa = max(squeeze(max(max(Xres((i-1)*mb+j, 1:end, 4)))-minval)); b = 1;
    zoQB = min(squeeze(min(min(Xres((i-1)*mb+j, end, 2)))), zoQB);
    timeQB = max(squeeze(max(max(Xres((i-1)*mb+j, 1:end, 3)))), timeQB);
    %Xres = dlmread(name);
    
%      semilogy(squeeze(hist1(end, end, 1:b:end)), squeeze(abs(hist1(i, j, 1:b:end) - minval(1:b:end)')), ...
%          char(marker((i-1)*m+j)), 'DisplayName',char(Legend((i-1)*m+j)), 'linewidth', 1.6,'markersize',4.5);
    
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
    
  end
 end
 method = '/SAGA/SAGA';
 scheme = 'SAGA';
 %mb = 1;
 for i=nB+1:nB+1
  for j=1:mb
%     Xres = [squeeze(hist1(end, end, 1:b:end)), squeeze(floor(zerothhist(i, j, 1:b:end)/1000)),...
%         squeeze(floor((timehist(i, j, 1:b:end)-timehist(i, j, 1))/1000)), ...
%           squeeze(abs(hist1(i, j, 1:b:end)))];
    %Xres = [squeeze(floor(zerothhist(i, j, 1:b:end)/1000)), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    %Xres = [squeeze(floor(timehist(i, j, 1:b:end))), squeeze(abs(hist1(i, j, 1:b:end))- minval)];
    Batch =  BatchArray(1, 1);
    batch_size = batch_sizeArray(1, j);
    Legend((i-1)*mb+j) = strcat(scheme, ' b = ', {' '}, num2str(batch_size));
    name = strcat('results/',par, method, 'B', num2str(1), 'b', num2str(j));
%     dlmwrite(name, Xres, 'delimiter', '\t', 'precision', 32);
    Xresinstant = dlmread(name, '\t');
    Xres((i-1)*mb+j,:, :) = Xresinstant;
    minval =  min(min(Xres((i-1)*mb+j, 1:end, 4)), minval);
    aa = max(squeeze(max(max(Xres((i-1)*mb+j, 1:end, 4)))-minval)); b = 1;
    zoQB = min(squeeze(min(min(Xres((i-1)*mb+j, end, 2)))), zoQB);
    timeQB = max(squeeze(max(max(Xres((i-1)*mb+j, 1:end, 3)))), timeQB);
    %Xres = dlmread(name);
    
%      semilogy(squeeze(hist1(end, end, 1:b:end)), squeeze(abs(hist1(i, j, 1:b:end) - minval(1:b:end)')), ...
%          char(marker((i-1)*m+j)), 'DisplayName',char(Legend((i-1)*m+j)), 'linewidth', 1.6,'markersize',4.5);
    
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
    
  end
 end
 
 % The properties we've been using in the figures
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);
 
for k=1:3
    h = figure(k);
 set(gcf,'position',[200,100,386,269]); 
     for i=nB:-1:1
  for j=1:mb
 
 semilogy(Xres((i-1)*mb+j,:,k), Xres((i-1)*mb+j,:,4)- minval, ...
        char(marker((i-1)*mb+j)));
    hold on;
  end 
     end
    for i=nB+1:nB+1
  for j=1:mb
  hold on;
 semilogy(Xres((i-1)*mb+j,:,k), Xres((i-1)*mb+j,:,4)- minval, ...
        char(marker((i-1)*mb+j)));
   
  end 
    end
hold off
switch k
    case 1
       xlabel('# of epochs');
       axis([0 300, 1E-12,aa])
    case  2 
       xlabel('Queries/1000');
%        axis([0 zoQB, 1E-5,aa]);
%        if(strcmp(par,'mnist'))
           axis([0 5e6, 0.5E-5,aa]);
%        end
       if(strcmp(par,'adult'))
           axis([0 2e6, 0.5E-5,aa]);
       end
       if(strcmp(par,'ijcnn1'))
           axis([0 1e5, 0.5E-5,aa]);
       end
    case  3 
       xlabel('CPU time (seconds)');
       axis([0 timeQB, 1E-6,aa]);
%        if(strcmp(par,'mnist'))
%            axis([0 6e4, 1E-5,aa]);
%        end
end

ylabel('Objective minus best');
% xlabel('Number of effective passes');
% ylabel('Objective minus best');
% axis([0 outer_loops, 1E-12,aa])
%legend('SAGA','SVRG','Katyusha','FSVRG');

legend(Legend, 'Location','southeast')
figname = strcat('results/', par, '/plot/', par, 'b', num2str(batch_sizeArray(1)), 'k', ...
    num2str(k),'.eps');
print(h,'-r300','-depsc', figname);
% export_fig(h, figname)
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
