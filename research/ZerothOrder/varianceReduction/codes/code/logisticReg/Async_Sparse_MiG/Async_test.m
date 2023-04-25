clear;
mex_all;
load 'rcv1_train.binary.mat';

%% Parse Data
X = [ones(size(X, 1), 1) X];
[N, Dim] = size(X);
X = X';

%% Normalize Data
sum1 = 1./sqrt(sum(X.^2, 1));
if abs(sum1(1) - 1) > 10^(-10)
    X = X.*repmat(sum1, Dim, 1);
end
clear sum1;

%% Set Params
passes = 300;
model = 'logistic'; % least_square / logistic
regularizer = 'L2';  
init_weight = repmat(0, Dim, 1); % Initial weight
lambda1 = 10^(-7); % L2. 
L = (0.25 * max(sum(X.^2, 1)) + lambda1);
sigma = lambda1;
is_sparse = issparse(X);
Mode = 1;
is_plot = true;
fprintf('Model: %s-%s\n', regularizer, model);

% Thread Number
thread_no = 8;

%% MiG
algorithm = 'AMiG';
theta = 0.3;
step_size = 1 / (0.5 * theta * L);
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
hist1 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1,...
    L, step_size, loop, is_sparse, sigma, thread_no, theta);
time = toc;
fprintf('Time: %f seconds \n', time);
X_MiG = [0:3:passes]';
hist1 = [X_MiG, hist1];

%% KroMagnon
algorithm = 'KroMagnon';
step_size = 1 / (0.4 * L);
loop = int64(passes / 3); % 3 passes per loop
fprintf('Algorithm: %s\n', algorithm);
tic;
hist2 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1,...
    L, step_size, loop, is_sparse, sigma, thread_no);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SVRG = [0:3:passes]';
hist2 = [X_SVRG, hist2];

%% ASAGA
algorithm = 'ASAGA';
step_size = 1 / (0.7 * L);
loop = int64((passes - 1) * N); % One Extra Pass for initialize SAGA gradient table.
fprintf('Algorithm: %s\n', algorithm);
tic;
hist3 = Interface(X, y, algorithm, model, regularizer, init_weight, lambda1,...
    L, step_size, loop, is_sparse, sigma, thread_no);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SAGA = [0 1 2:3:passes - 2]';
hist3 = [X_SAGA, hist3];

%% Plot
if(is_plot)
    optimal = 0.013298006616553; % For L2-logistic on RCV1-train, lambda = 10^(-7)
    minval = optimal - 2e-16;
    aa = max(max([hist2(:, 2)])) - minval; 
    b = 3;

    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'m--d','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'b--+','linewidth',1.2,'markersize',4.5);
    hold on,semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2) - minval),'k-.o','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of effective passes');
    ylabel('Objective minus best');
    axis([0 passes, 1E-12, aa]);
    legend('KroMagnon', 'ASAGA', 'MiG');
end
