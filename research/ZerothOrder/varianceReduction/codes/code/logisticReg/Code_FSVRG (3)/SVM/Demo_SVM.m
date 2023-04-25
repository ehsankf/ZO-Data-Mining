% Demo for Support Vector Machine (SVM)

clear all; 
close all;

mexinstall;

epochs = 5;
runs   = 10; % 100 

%%% SSGD
opt_method = 'SSGD';  % Stochastic Sub-Gradient Descent (SSGD)

accuracy1 = zeros(runs,1);
for i = 1:runs
    [accuracy1(i),time1(i)] = Classification(opt_method, epochs);
end
acc1 = mean(accuracy1)
std1 = std(accuracy1)


%%% SVRSG
opt_method = 'SVRSG';  % Stochastic Variance Reduced Gradient (SVRSG) 

accuracy2 = zeros(runs,1);
for i= 1:runs
    [accuracy2(i),time2(i)] = Classification(opt_method, epochs);
end
acc2 = mean(accuracy2)
std2 = std(accuracy2)


%%% FSVRG
opt_method = 'FSVRG';  % Fast Stochastic Variance Reduced Gradient (FSVRG)
accuracy3 = zeros(runs,1);
for i= 1:runs
    [accuracy3(i),time3(i)] = Classification(opt_method, epochs);
end
acc3 = mean(accuracy3)
std3 = std(accuracy3)


%%% Results
b  = bar([acc1,acc2,acc3]);
ch = get(b,'children'); 
set(ch,'FaceVertexCData',[1;2;3])
set(gca,'XTickLabel',{'SSGD','SVRSG','FSVRG'})
axis([0.45 3.55 0.88 0.92])
ylabel('Testing accuracy %')


