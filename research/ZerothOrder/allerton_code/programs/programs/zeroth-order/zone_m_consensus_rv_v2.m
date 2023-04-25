% Zones for consensus problem
%%
clc;
clear;
close all;

%% Parameters
max_trial = 50; % max # of trials
max_iter = 1000; % # of iterations per trial
M = 1;  % problem dimention
Radius = 0.6;
N = 20; % number of agents
seed=1;  % for random number generators
I_N = eye(N,N);
fig = 1; % 1:plot the results, 0: no plot
m = 10; % number of gradient estimations
%% functions
fc = @(a,b,x,xi) a / (1 + exp(-x))+ b * log(1+x^2)+xi;             % functional value
gc = @(a,b,x) a * exp(-x)/((1+exp(-x))^2) + b*2*x/((1+x^2)); % gradient value

%% Network Topology Generation
while (1)
    [error,Adj, degree, xy, Lmatrix]=randomgraph(N,Radius,seed);
    if ~error
        break;
    end
end

Deg_Matrix = diag(degree); % diagonal degree matrix
E = sum(sum(Adj))/2;
A = zeros(E,N);
edge_index = 1;
for ii = 1 : N
    for jj = 1 : N
        if (jj>ii)
            if Adj(ii,jj) == 1
                A(edge_index,jj) = 1;
                A(edge_index,ii) = -1;
                edge_index = edge_index + 1;
            end
        end
    end
end

PW = zeros(N,N); % Metropolic-weight matrix
for ii = 1 : N
    for jj = ii+1 : N
        if Adj(ii,jj) == 1
            PW(ii,jj) = 1/(1+max(degree(ii), degree(jj)));
            PW(jj,ii) = PW(ii,jj);
        end
    end
    PW(ii,ii) = 1-sum(PW(ii,:));
end

B = abs(A);
Lp = B' * B;
Lm = A' * A;
D = Deg_Matrix;
inv_D = inv(D);
A_bar = I_N-inv_D*Lm/2;
ME = edge_index-1;
lambda = zeros(ME,max_iter);
y_temp = zeros(N,max_iter);
eig_Lm = eig(Lm);
for ii = 1 : length(eig_Lm)
    if (eig_Lm(ii)>=1e-10)
        min_eig_Lm = eig_Lm(ii);
        break;
    end
end

object_value11 = zeros(max_iter-1,max_trial);
object_value12 = zeros(max_iter-1,max_trial);
Opt_compare11 = zeros(max_iter-1,max_trial);
Opt_compare12 = zeros(max_iter-1,max_trial);
Constraint_vio11 = zeros(max_iter-1,max_iter);
Constraint_vio12 = zeros(max_iter-1,max_iter);
P_RGF21 = zeros(max_iter-1,max_trial);
Opt_compare21 = zeros(max_iter-1,max_trial);
Constraint_vio21 = zeros(max_iter-1,max_iter);


for trial = 1 : max_trial
    trial
    a0 = abs(randn(N,1));
    a = a0/norm(a0,'fro'); %normalize data
    b0 = abs(randn(N,1));
    b = b0/norm(b0,'fro'); % normalize data
    y_temp(:,1) = randn(N,1);
    %% ZONE-S algorithm
    L1 = zeros(1,N);
    for ii = 1 : N
        L1(ii) = a(ii) * 0.09622 + b(ii) *5;
    end
    
    big_L = max(L1);
    ah = 2+6*norm(Lp,'fro');
    bh = -big_L*(big_L+24*norm(Lp,'fro')/min_eig_Lm+1)-3;
    dh = -12*big_L/min_eig_Lm;
    root = (-bh+sqrt(bh^2-4*ah*dh))/(2*ah);
    rho = max(big_L/2,root);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % run with constant step size: L;
    gradient1 = zeros(1,max_iter);
    Opt_gap = zeros(1,max_iter);
    P = zeros(1,max_iter);
    x = y_temp;
    lambda = zeros(edge_index-1,max_iter);
    mu = 1/sqrt(max_iter);  % smoothing parameter
    % main loop
    while (1)
        for iter  = 2 : max_iter
            gradient_zo = zeros(N,1); % zeroth-order gradient
            gradient_ex = zeros(N,1); % exact gradient
            for ii = 1 : N
                % calculate zero-order gradient
                g0 = 0;
                for jj=1:m
                    phi = normrnd(0,1);
                    xi = normrnd(0,0.01);
                    f1 = fc(a(ii),b(ii),x(ii,iter-1) + mu*phi,xi);
                    f2 = fc(a(ii),b(ii),x(ii,iter-1),xi);
                    g0 = g0 + 1/mu*(f1 - f2)*phi;
                end
                gradient_zo(ii) = g0/m; % zero-order gradient (average over m0)
                gradient_ex(ii) = gc(a(ii),b(ii),x(ii,iter-1));
            end
            
            x(:,iter) = A_bar*x(:,iter-1) - inv_D/(2*rho)* (gradient_zo + A.'*lambda(:,iter-1)); % primal update
            lambda(:,iter) = lambda(:,iter-1) + rho * (A*x(:,iter)); % due update
            
            
            Constraint_vio11(iter-1,trial) =  norm(A*x(:,iter))^2;
            %Opt_compare11(iter-1,trial) = norm(gradient_ex)^2 + Constraint_vio11(iter-1,trial);
            Opt_compare11(iter-1,trial) = (norm(sum(gradient_ex))^2 + Constraint_vio11(iter-1,trial))/(1+size(A*x(:,iter),1) * size(A*x(:,iter),2)); %scaled
        end
        if (iter == max_iter)
            break;
        end
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ZONE-m run with rho = (iter)^(1/2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gradient1 = zeros(1,max_iter);
    Opt_gap = zeros(1,max_iter);
    P = zeros(1,max_iter);
    x = y_temp;
    lambda = zeros(edge_index-1,max_iter);
    mu = 1/sqrt(max_iter);  % smoothing parameter
    % main loop
    while (1)
        for iter  = 2 : max_iter
            rho = .07*sqrt(iter);
            gradient_zo = zeros(N,1);
            gradient_ex = zeros(N,1);
            for ii = 1 : N
                % calculate zero-order gradient
                g0 = 0;
                for jj=1:m
                    phi = normrnd(0,1);
                    xi = normrnd(0,0.01);
                    f1 = fc(a(ii),b(ii),x(ii,iter-1) + mu*phi,xi);
                    f2 = fc(a(ii),b(ii),x(ii,iter-1),xi);
                    g0 = g0 + 1/mu*(f1 - f2)*phi;
                end
                gradient_zo(ii) = g0/m; % zero-order gradient (average over m0)
                gradient_ex(ii) = gc(a(ii),b(ii),x(ii,iter-1));
            end
            x(:,iter) = A_bar*x(:,iter-1) - inv_D/(2*rho)* (gradient_zo + A.'*lambda(:,iter-1)); % primal update
            lambda(:,iter) = lambda(:,iter-1) + rho * (A*x(:,iter)); % due update
            
            Constraint_vio12(iter-1,trial) =  norm(A*x(:,iter))^2;
            %Opt_compare11(iter-1,trial) = norm(gradient_ex)^2 + Constraint_vio11(iter-1,trial);
            Opt_compare12(iter-1,trial) = (norm(sum(gradient_ex))^2 + Constraint_vio12(iter-1,trial))/(1+size(A*x(:,iter),1) * size(A*x(:,iter),2)); %scaled
        end
        if (iter == max_iter)
            break;
        end
    end
    
    %% Randomized Gradient-free (RGF) Deming Yuan et al IEEE Transaction on neural networks and learning sustems
    
    x = y_temp;
    gradient2_zo = zeros(1,max_iter);
    gradient2 = zeros(1,max_iter);
    for iter  = 2 : max_iter
        gamma = 5/sqrt(iter);

        mu = 1e-4;
        for ii = 1 : N
            x(ii,iter) = 0;
            for jj = 1 : N
                if (Adj(ii,jj) == 1)
                    x(ii,iter) = x(ii,iter) +  1/(sum(Adj(ii,:))) * x(jj,iter-1);
                end
            end
            g0 = 0;
            for jj=1:m
                phi = normrnd(0,1);
                xi = normrnd(0,0.01);
                f1 = fc(a(ii),b(ii),x(ii,iter-1) + mu*phi,xi);
                f2 = fc(a(ii),b(ii),x(ii,iter-1),xi);
                g0 = g0 + 1/mu*(f1 - f2)*phi;
            end
            gra2_zo = g0/m; % zero-order gradient
            x(ii,iter) = x(ii,iter) - gamma * gra2_zo;
        end
        for ii = 1 : N
            xi = normrnd(0,0.01);
            P_RGF21(iter-1,trial) = P_RGF21(iter-1,trial) + fc(a(ii),b(ii),x(ii,iter),xi);
        end
        for ii = 1 : N
            gradient2_zo(iter-1) = gradient2_zo(iter-1) + gra2_zo;
            gradient2(iter-1) = gradient2(iter-1) + gc(a(ii),b(ii),x(ii,iter-1));
        end
        Constraint_vio21(iter-1,trial) =  norm(A*x(:,iter),'fro')^2;
        %Opt_compare21(iter-1,trial) = norm(gradient_ex)^2 + Constraint_vio21(iter-1,trial);
        Opt_compare21(iter-1,trial) = (norm(gradient2(iter-1),'fro')^2 + norm(A*x(:,iter),'fro')^2)/(1+size(A*x(:,iter),1) * size(A*x(:,iter),2));
    end
end
% print the result
opt_gap_ZENITH = mean(Opt_compare11(max_iter-1,:));
opt_gap_ZENITH_I = mean(Opt_compare12(max_iter-1,:));
opt_gap_DSG = mean(Opt_compare21(max_iter-1,:));
cons_vio_ZENITH = mean(Constraint_vio11(max_iter-1,:));
cons_vio_ZENITH_I = mean(Constraint_vio12(max_iter-1,:));
cons_vio_DSG = mean(Constraint_vio21(max_iter-1,:));
%% plot the results
if (fig==1)
    
    fontsize = 14;
    linewidth = 2;
    linewidth2 = 3;
    MarkerSize = 4;
    %% optimality gap
    figure;
    semilogy(mean(Opt_compare11,2), '-','linewidth',linewidth);hold on;
    semilogy(mean(Opt_compare12,2), '--','linewidth',linewidth);hold on;
    semilogy(mean(Opt_compare21,2), '-','linewidth',linewidth2);
    le = legend('ZONE-M  $$\rho=$$ constant','ZONE-M $$\rho = \sqrt{r}$$','RGF step-size = $$\frac{1}{\sqrt{r}}$$');
    xl = xlabel('Iteration Number','FontSize',fontsize);
    yl = ylabel('Optimality Gap','FontSize',fontsize);
    grid on;
    set(le,'Interpreter','latex');
    set(xl,'Interpreter','latex');
    set(yl,'Interpreter','latex');
    set(gca,'FontSize',fontsize);
    set(gcf,'DefaultTextFontSize',fontsize);
    print(gcf,'-r300','-depsc','figs/opt_gap_cons.eps');
    savefig('figs/opt_gap_cons.fig');
    
    %% contraint violation
    figure;
    semilogy(mean(Constraint_vio11,2),'-', 'linewidth',linewidth); hold on;
    semilogy(mean(Constraint_vio12,2),'--','linewidth',linewidth);hold on;
    semilogy(mean(Constraint_vio21,2), '-','linewidth',linewidth2);hold on;
    le = legend('ZONE-M  $$\rho=$$ constant','ZONE-M $$\rho = \sqrt{r}$$','RGF step-size = $$\frac{1}{\sqrt{r}}$$');
    xl = xlabel('Iteration Number','FontSize',fontsize);
    yl = ylabel('Consensus Error','FontSize',fontsize);
    set(le,'Interpreter','latex');
    set(xl,'Interpreter','latex');
    set(yl,'Interpreter','latex');
    grid on;
    set(gca,'FontSize',fontsize);
    set(gcf,'DefaultTextFontSize',fontsize);
    print(gcf,'-r300','-depsc','figs/cons_vio_cons.eps');
    savefig('figs/cons_vio_cons.fig');
    
end
file_name = 'consensus_result.mat';
save(file_name);
