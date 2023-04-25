% Show results
clc; clear;
close all;

width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 4;

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
 
h1 = figure(1)
set(gcf,'position',[200,100,386,269]); 
h2 = figure(2)
set(gcf,'position',[200,100,386,269]); 
% set(h1,'position',[200,100,386,269]) 
% set(h2,'position',[200,100,386,269]) 
marker = {'-', 'g-', 'r-', 'b-'};
nB = 4;
Legend = {'ZO-ProxSGD','ZO-ProxSVRG', 'ZO-ProxSVRG (RandSGE)', 'ZO-ProxSAGA'};
% Dir = {'../../ZOSVRG-BlackBox-Adv-mastercopy/Results/ZOSVRG', 'ZOSAGA'};
Dir = {'../../ZOSVRG-BlackBox-Adv-mastercopy/Results/ZOSVRG'}
Dir = {'../../ZOSVRG-BlackBox-Adv-masterSGDV1/Results/ZOSGD'}
Dir = {'../../ZOSVRG-BlackBox-Adv-masterSGDV1/Results/ZOSGD', ...
    '../../ZOSVRG-BlackBox-Adv-master/Results/ZOSVRG' , '../../RandSGE/Results/ZOSVRG', ...
    '../../ZOSVRG-BlackBox-Adv-masterSAGA/Results/ZOSAGA'}
%Dir = {'ZOSAGA'}
for i=1:nB
    dirpath = cell2mat(strcat(Dir(i), '/log.xlsx'));
    Xres = rmmissing(readtable(dirpath, 'Range','18:600000'));
    ms = size(Xres, 1);
    %     Xplot = [Xres{20:ms,6}/1000, Xres{20:ms,12}];
    if (i == 1)
       ms = 10000
       Xplot = [Xres{20:ms,5}/1000, Xres{20:ms,11}]; 
       Xplot1 = [Xres{20:ms,3}, Xres{20:ms,11}];
    else
        Xplot = [Xres{20:ms,6}/1000, Xres{20:ms,12}];
        Xplot1 = [(Xres{20:ms,2}-1)+Xres{20:ms,4}, Xres{20:ms,12}];
    end
    Xplot(1,:) = [0, 23];
    Xplot1(1,:) = [0, 23];
%     if (nB == 2)
%        Zplot = find(Xplot(1:end,1) > 500 & Xplot(1:end,2) > 3);
%        Xplot(Zplot, :) = [];
%     end
    figure(1)
    plot(Xplot(:,1), Xplot(:,2), ...
        char(marker(i)), 'DisplayName',char(Legend(i)), 'linewidth', 1.6,'markersize',4.5);
    hold on;
    figure(2)
    plot(Xplot1(:,1), Xplot1(:,2), ...
        char(marker(i)), 'DisplayName',char(Legend(i)), 'linewidth', 1.6,'markersize',4.5);
%      semilogy(squeeze(hist1(end, end, 1:b:end)), squeeze(abs(hist1(i, j, 1:b:end) - minval(1:b:end)')), ...
%          char(marker((i-1)*m+j)), 'DisplayName',char(Legend((i-1)*m+j)), 'linewidth', 1.6,'markersize',4.5);
    hold on;
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
end
figure(1)
hold off
xlabel('Queries/1000');
ylabel('Black-box attack loss');
axis([0 50000 0 22]);
%axis([0 zoQB/1000, 1E-12,aa])
%xlabel('Number of effective passes');
% ylabel('Objective minus best');
% axis([0 outer_loops, 1E-12,aa])
legend(Legend);
%legend(Legend)
saveas(h1,'figiter.pdf')
figure(2)
hold off
xlabel('# of iterations');
ylabel('Black-box attack loss');
axis([0 3000 0 22]);
%axis([0 zoQB/1000, 1E-12,aa])
%xlabel('Number of effective passes');
% ylabel('Objective minus best');
% axis([0 outer_loops, 1E-12,aa])
legend(Legend);
%legend(Legend)
saveas(h2,'figquery.pdf')