% Show results
clc; clear;
h = figure(11)
set(gcf,'position',[200,100,386,269]) 
marker = {'-', 'g-', 'r-', 'b-'};
nB = 2;
Legend = {'ZO-ProxSGD','ZO-ProxSVRG','ZO-ProxSAGA','ZO-ProxPSCVR'};
Dir = {'../../ZOSVRG-BlackBox-Adv-mastercopy/Results/ZOSVRG', 'ZOSAGA'};
for i=1:nB
    dirpath = cell2mat(strcat(Dir(i), '/log.xlsx'));
    Xres = rmmissing(readtable(dirpath, 'Range','18:600000'));
    ms = size(Xres, 1);
    Xplot = [Xres{20:ms,6}/1000, Xres{20:ms,12}];
%     if (nB == 2)
%        Zplot = find(Xplot(1:end,1) > 500 & Xplot(1:end,2) > 3);
%        Xplot(Zplot, :) = [];
%     end
    plot(Xplot(:,1), Xplot(:,2), ...
        char(marker(i)), 'DisplayName',char(Legend(i)), 'linewidth', 1.6,'markersize',4.5);
%      semilogy(squeeze(hist1(end, end, 1:b:end)), squeeze(abs(hist1(i, j, 1:b:end) - minval(1:b:end)')), ...
%          char(marker((i-1)*m+j)), 'DisplayName',char(Legend((i-1)*m+j)), 'linewidth', 1.6,'markersize',4.5);
    hold on;
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'b:d','linewidth',2.6,'markersize',4.5); 
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'r--^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist4(1:b:end,1), abs(hist4(1:b:end,2) - minval),'k-p','linewidth',1.2,'markersize',4.5); 
end
hold off
xlabel('Queries/1000');
ylabel('Black-box attack box');
%axis([0 zoQB/1000, 1E-12,aa])
%xlabel('Number of effective passes');
% ylabel('Objective minus best');
% axis([0 outer_loops, 1E-12,aa])
legend('SAGA','SVRG','Katyusha','FSVRG');
%legend(Legend)
saveas(h,'fig11.pdf')