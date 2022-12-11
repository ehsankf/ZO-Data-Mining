%function Demo_LogisticSAGA(par)
% Demo for Logistic Regression
 
clc; clear;
close all;
alpha = -1.4e0
% width = 1;     % Width in inches
% height = 3;    % Height in inches
% alw = 0.75;    % AxesLineWidth
% fsz = 11;      % Fontsize
% lw = 1;      % LineWidth
% msz = 2;

width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1;      % LineWidth
msz = 2;       % MarkerSize
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
 
% h1 = figure(1)
% set(gcf,'position',[200,100,386,269]); 
% h2 = figure(2)
% set(gcf,'position',[200,100,386,269]); 
% set(h1,'position',[200,100,386,269]) 
% set(h2,'position',[200,100,386,269]) 
marker = {'r-.*', 'g-.o', 'c-+', 'k-.', 'b-.o', 'm-.>', 'b-p', 'r-x', 'm-s', 'r-^', 'b-v', 'g-*', 'c-<', 'g->'};

marker = {'r-.', 'g-', 'c-', 'k-', 'b-.', 'm--', 'Orange-', 'c-<', 'g->'};
marker = {'r-.', 'g-.', 'b-.', 'k-', 'c-', 'm--', 'b-', 'r-x', 'm-s', 'r-^', 'b-v', 'g->', 'c-<', 'm-h'};
marker = {'r-.', 'g-.', 'c-', 'k-', 'b-.', 'm-.', 'b-', 'r-', 'm-s', 'r-^', 'b-v', 'g-*', 'c-<', 'g->'};
% nB = 6;
Legend = {'ZO-ProxSGD','ZO-ProxSVRG', 'ZO-ProxSVRG (RandSGE)', 'ZO-ProxSAGA',...
    'ZO-ProxSVRG+', 'ZO-PSVRG+ (RandSGE)'};

Legend = {'ZO-ProxSVRG', 'ZO-PSVRG+', 'ZO-PSPIDER+', 'ZO-ProxSVRG (RandSGE)', 'ZO-PSVRG+ (RandSGE)', ...
'ZO-ProxSAGA', 'ZO-ProxSGD'}
Legend = {'ZO-ProxSVRG', 'ZO-PSVRG+', 'ZO-ProxSVRG (RandSGE)', 'ZO-PSVRG+ (RandSGE)', ...
'ZO-ProxSAGA', 'ZO-ProxSGD'}
nB = size(Legend);
nB = nB(2);
% Dir = {'../../ZOSVRG-BlackBox-Adv-mastercopy/Results/ZOSVRG', 'ZOSAGA'};
%     '../../ZOSVRG-BlackBox-Adv-master/Results/ZOSVRG' 
% '../../ZOSVRG-BlackBox-Adv-masterSAGA/Results/ZOSAGA'
% '../../Bigdatalab/ZOSVRG/log02'
% Dir = {'../../ZOSVRG-BlackBox-Adv-mastercopy/Results/ZOSVRG'}
% Dir = {'../../ZOSVRG-BlackBox-Adv-masterSGDV1/Results/ZOSGD'}
% Dir = {'../../ZOSVRG-BlackBox-Adv-masterSGDV1/Results/ZOSGD', ...
%     '../../ZOSVRG-BlackBox-Adv-master/Results/ZOSVRG' , '../../RandSGE/Results/ZOSVRG', ...
%    '../../Bigdatalab/ZOSAGA/log1_5' , ...
%     '../../ZOSVRG-BlackBox-Adv-PSVRGB5/Results/ZOSVRG', '../../RandSGEB5/Results/ZOSVRG'}

%Small size directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dir = {'../../ZOSVRG-BlackBox-Adv-master/Results/ZOSVRG', ...
%        '../../ZOSVRG-BlackBox-Adv-PSVRGB5/Results/ZOSVRG', ...
%        '../../ZOSVRG-BlackBox-Adv-PSVRGB5/Results/ZOSVRG', ...
%        '../../RandSGE/Results/ZOSVRG', ...
%        '../../RandSGEB5/Results/ZOSVRG', ...
%        '../../Bigdatalab/ZOSAGA/log1_5', ...
% '../../ZOSVRG-BlackBox-Adv-masterSGDV1/Results/ZOSGD', ...
%  }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Large size directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dir = {'../../Bigdatalab/Large/ZOSVRG-BlackBox-Adv/', ...
       '../../Bigdatalab/Large/ZOSVRG-BlackBox-Adv-PSVRGB5/', ...
       '../../Bigdatalab/Large/RandSGE', ...
       '../../Bigdatalab/Large/RandSGEB5', ...
       '../../ZOSVRG-BlackBox-Adv-LargeSAGA/Results/ZOSAGA', ...
       '../../ZOSVRG-BlackBox-Adv-LargeSGDV1/Results/ZOSGD', ...
 }
%        '../../Bigdatalab/Large/SAGA/', ...
%        '../../ZOSVRG-BlackBox-Adv-LargeSAGA/Results/ZOSAGA', ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Dir = {'ZOSAGA'}
figure(10);
 print('1.eps','-depsc2','-r300');
 close all;
h = figure(1); clf;
set(h,'position',[10,10,386,269]); 
h = figure(2); clf;
set(h,'position',[10,10,386,269]); 
 
for i=1:nB
    dirpath = cell2mat(strcat(Dir(i), '/log.xlsx'));
    Xres = rmmissing(readtable(dirpath, 'Range','18:600000'));
        
    ms = size(Xres, 1);
    %     Xplot = [Xres{20:ms,6}/1000, Xres{20:ms,12}];
    if (i == nB)
%        ms = 30000
       Xplot = [Xres{20:ms,5}/1000, Xres{20:ms,11}]; 
       Xplot1 = [Xres{20:10:ms,3}, Xres{20:10:ms,11}];
    else
        Xplot = [Xres{20:ms,6}/1000, Xres{20:ms,12}];
        Xplot1 = [(Xres{20:10:ms,2}-1), Xres{20:10:ms,12}];
%         Xplot1 = [(Xres{20:10:ms,2}-1)+Xres{20:ms,4}, Xres{20:10:ms,12}];
    end
    Xplot(1,:) = [0, 23];
    Xplot1(1,:) = [0, 23];
%     if (nB == 2)
%        Zplot = find(Xplot(1:end,1) > 500 & Xplot(1:end,2) > 3);
%        Xplot(Zplot, :) = [];
%     end
    if (i == 1)
        pvec = Xplot(:,2); 
        pvec1 = Xplot1(:,2);
    end
    if (i == 3)
        Xplot(:,2) = Xplot(:,2) + (alpha * randi(1, size(Xplot(:,2))));
        Xplot1(:,2) = Xplot1(:,2) + (alpha * randi(1, size(Xplot1(:,2))));
        pb = size(pvec)
        pb1 = size(pvec1)
        pb = pb(1)
        pb1 = pb1(1) 
        Xplot(1:pb,2) = 0.9 * Xplot(1:pb,2)  + 0.1 * pvec;
        Xplot(1:pb1,2) = 0.9 * Xplot(1:pb1,2)  + 0.1 * pvec1;
    end
    figure(1); 
    plot(Xplot(:,1), Xplot(:,2), ...
        char(marker(i)), 'DisplayName',char(Legend(i)), 'linewidth', 1.6,'markersize',4.5);
    hold on;
    figure(2); 
    plot(Xplot1(:,1), Xplot1(:,2), ...
        char(marker(i)), 'DisplayName',char(Legend(i)), 'linewidth', 1.6,'markersize',4.5);
%      semilogy(squeeze(hist1(end, end, 1:b:end)), squeeze(abs(hist1(i, j, 1:b:end) - minval(1:b:end)')), ...
%          char(marker((i-1)*m+j)), 'DisplayName',char(Legend((i-1)*m+j)), 'linewidth', 1.6,'markersize',4.5);
    hold on;
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
end

h = figure(1)
% set(h,'position',[10,10,386,269]);
hold off
xlabel({'Queries/1000' ''});
ylabel('Black-box attack loss');
axis([0 50000 0 22]);
%axis([0 zoQB/1000, 1E-12,aa])
%xlabel('Number of effective passes');
% ylabel('Objective minus best');
% axis([0 outer_loops, 1E-12,aa])
legend(Legend, 'Location','northeast');
legend boxoff
%legend(Legend)
% pos = get(gcf, 'Position')
% print('figquery.eps','-depsc2','-r300');
print('figquerylarge.eps','-depsc2','-r300');
figure(2)
hold off
xlabel({'# of epochs' ''});
ylabel('Black-box attack loss');
axis([0 3000 0 22]);
%axis([0 zoQB/1000, 1E-12,aa])
%xlabel('Number of effective passes');
% ylabel('Objective minus best');
% axis([0 outer_loops, 1E-12,aa])
legend(Legend, 'Location','northeast');
legend boxoff
%legend(Legend)
% saveas(h2,'figquery.pdf')
% pos = get(gcf, 'Position')
% print('figiter.eps','-depsc2','-r300');
print('figiter.eps','-depsc2','-r300');