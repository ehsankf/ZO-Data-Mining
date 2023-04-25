
% distributed binary classification mini-batch with nonsmooth
% regularization lam\|x\|_1 using inexact prox_pda with inexact cocensus 
% Davood, May 12, 2017
clc;
clear;
close all;

tol = 1e-4;
%% Parameters
N = 20; % number of agents in the network
n = 10;  % problem dimention
bs = 10; % batch size
K = bs*N; % number of data points
max_trial = 2; % number of trials
max_iter = 2000; % number of iterations per trial
beta = 0.001;
alpha = 1;
zeta = 1e+4;
%% functions
fc = @(x,beta,alpha,z,y,N) 1/N*log(1+exp(-y*x.'*z))+beta/N * sum(alpha*x.^2./(1+alpha*x.^2)); % for evaluting the smooth objective value
gc = @(x,beta,alpha,z,y,N) 1/N*exp(-y*x.'*z)*(-y * z)/(1+exp(-y*x.'*z))+beta/N*(2*alpha*x.*(1+alpha*x.^2)-alpha*x.^2.*(2*alpha*x))./((1+alpha*x.^2).^2);
soft= @(z,lam) (abs(z)>=lam).*(abs(z)-lam).*sign(z);

Radius = .7;
seed=1;  % for random number generators

%% Network Topology Generation
while (1)
    [error,Adj, degree, xy, Lmatrix]=randomgraph(N,Radius,seed);
    if ~error
        break;
    end
end


Deg = diag(degree); %% diagonal degree matrix

E = sum(sum(Adj))/2;
A = zeros(E,N);
edge_index = 1;
% incidence matrix
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

A = kron(A,eye(n));
C = [A'*A -A';-A eye(E*n,E*n)];
l_max = eigs(C,1);
B = abs(A);
Lp = B' * B; % singless Laplacian
Lm = A' * A; % signed LAplacian
b_tilde = 0;

Opt_gap1 = zeros(max_iter-1,max_trial);
Opt_gap2 = zeros(max_iter-1,max_trial);
Constraint_vio1 = zeros(max_iter-1,max_iter);
Constraint_vio2 = zeros(max_iter-1,max_iter);

for trial = 1 : max_trial
    trial
    x_temp = randn(N*n,max_iter);
    x_temp(:,1) = rand(N*n,1);
    z = zeros(n*E,1);
    features = randn(n,K);
    features_norm = features/norm(features,'fro'); % data normalization
    big_L = 1/K*norm(features_norm,'fro')^2;
    labels = randi([1,2], 1, K); labels(find(labels==2)) = -1; % labels \in {-1,1}
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% Prog-GPDA Constant step_size %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Parameters
    option = 'op1';
    switch option
        case 'op1'
            lambda = 0.000; % sparsity parameter
            eps = 1e-2; % epsilon error
            gam = eps; % inexact parameter
            %gam = 0;
            rho =  .002/gam; % penalty parameter
            %            beta = 2*bigL; % proximal term factor
            %rho = 0.02;

    end
    x = x_temp;
    mu = zeros((edge_index-1)*n,max_iter);
    x_matrix = zeros(n,N);
    for iter  = 2:max_iter
        
        % calculate the gradient
        gradient = zeros(N*n,1);
        d = A*x(:,iter-1)- z +(1-rho*gam)/rho*mu(:,iter-1);
        
        gradient_matrix = zeros(n,N);
        for ii = 1 : N
            for jj=(ii-1)*bs+1:ii*bs
                gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x((ii-1)*n+1:ii*n,iter-1),beta, alpha, features(:,jj), labels(jj),K);
            end
            gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
        end
        % update x,
       
        x(:,iter) = x(:,iter-1) - gradient/(rho*l_max) - A'*d/l_max;
        x_matrix = reshape(x(:,iter),n,N);
        z_old = z;
        z = z + d/l_max;


% project into the box
for j=1:E
    z_help = z((j-1)*n+1:j*n);
    for k=1:n
        if z_help(k)>tol
            z_help(k) = tol;
        end
        if z_help(k)<-tol
            z_help(k) = -tol;
        end
    end
    z((j-1)*n+1:j*n) = z_help;
end
    
        %update due
        mu(:,iter) = (1-rho*gam)*mu(:,iter-1) + rho * (A*x(:,iter)-z);
        
       
        w1 = gradient + A'*mu(:,iter);
        w2 = z + mu(:,iter);


for j=1:E
    w2_help = w2((j-1)*n+1:j*n);
    for k=1:n
        if w2_help(k)>tol
            w2_help(k) = tol;
        end
        if w2_help(k)<-tol
            w2_help(k) = -tol;
        end
    end
    w2((j-1)*n+1:j*n) = w2_help;
end

        w = [w1;z-w2];
        DT1(iter-1,trial) = norm(w1)^2;
               
        Opt_gap1(iter-1,trial) =  norm([w1;z-w2])^2 + norm(A*x(:,iter)-z)^2;
        Constraint_vio1(iter-1,trial) =  (norm(A*x(:,iter), 'inf'));
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%% Prox-GPDA step_size=iter %%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    x = x_temp;
    mu = zeros((edge_index-1)*n,max_iter);
    x_matrix = zeros(n,N);
    for iter  = 2:max_iter
        rho = .001*sqrt(iter);
        gam = 1/(iter);
        % calculate the gradient
        gradient = zeros(N*n,1);
        d = A*x(:,iter-1)- z +(1-rho*gam)/rho*mu(:,iter-1);
        
        gradient_matrix = zeros(n,N);
        for ii = 1 : N
            for jj=(ii-1)*bs+1:ii*bs
                gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x((ii-1)*n+1:ii*n,iter-1),beta, alpha, features(:,jj), labels(jj),K);
            end
            gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
        end
        % update x,
       
        x(:,iter) = x(:,iter-1) - gradient/(rho*l_max) - A'*d/l_max;
        x_matrix = reshape(x(:,iter),n,N);
        z_old = z;
        z = z + d/l_max;
        
for j=1:E
    z_help = z((j-1)*n+1:j*n);
    for k=1:n
        if z_help(k)>tol
            z_help(k) = tol;
        end
        if z_help(k)<-tol
            z_help(k) = -tol;
        end
    end
    z((j-1)*n+1:j*n) = z_help;
end

        %update due
        mu(:,iter) = (1-rho*gam)*mu(:,iter-1) + rho * (A*x(:,iter)-z);
        
       
        w1 = gradient + A'*mu(:,iter);
        w2 = z + mu(:,iter);
for j=1:E
    w2_help = w2((j-1)*n+1:j*n);
    for k=1:n
        if w2_help(k)>tol
            w2_help(k) = tol;
        end
        if w2_help(k)<-tol
            w2_help(k) = -tol;
        end
    end
    w2((j-1)*n+1:j*n) = w2_help;
end

        w = [w1;z-w2];
        
              
        Opt_gap2(iter-1,trial) =  norm([w1;z-w2])^2 + norm(A*x(:,iter)-z)^2;
        DT2(iter-1,trial) = norm([(gradient+(1-rho*gam)*A'*mu(:,iter)+rho*A'*(A*x(:,iter)-z))/zeta;(z-w2)])^2;
        
        %%% calculate potential function
        Constraint_vio2(iter-1,trial) =  (norm(A*x(:,iter), 'inf'));
    end
   
end
%% plot the results
linewidth = 2.5;
fontsize = 17;
MarkerSize = 16;
leg_size = 15;

figure;
% plot all trials
for trial = 1:max_trial
    semilogy(Opt_gap1(:,trial),'--','color',[1,0.7,0.7]);hold on;
    semilogy(Opt_gap2(:,trial),'--', 'color',[.5,0.7,1]);hold on;
    %semilogy(Opt_gap_dsg(:,trial),'--', 'color',[.8,1,0.5]);hold on;
end

 semilogy(mean(Opt_gap1,2),'linewidth',linewidth,'color', 'r');hold on;
YVec = mean(Opt_gap1,2);
XVec = 1:length(YVec);
p1 = semilogy(XVec(300:200:end),YVec(300:200:end), '-+','linewidth',linewidth, 'color', 'r'); hold on;

semilogy(mean(Opt_gap2,2),'linewidth',linewidth,'color', 'b');
YVec = mean(Opt_gap2,2);
XVec = 1:length(YVec);
p2 = semilogy(XVec(100:200:end),YVec(100:200:end), '-*','linewidth',linewidth, 'color', 'b'); hold on;

le = legend([p1,p2],{'PProx-PDA $\rho=$ constant','PProx-PDA-IA $\rho = \mathcal{O}(r)$'},'FontSize',leg_size);
xl = xlabel('Iteration Number','FontSize',fontsize);
yl = ylabel('Stationarity  Gap','FontSize',fontsize);
%grid on;
set(gca,'FontSize',fontsize);
set(le,'Interpreter','latex');
set(xl,'Interpreter','latex');
set(yl,'Interpreter','latex');

savefig('figs/ppda_log_reg_opt_gap.fig');
print(gcf,'-r300','-depsc','figs/ppda_log_reg_opt_gap.eps');

figure;
semilogy(mean(Constraint_vio1,2), 'linewidth',linewidth);hold on;
semilogy(mean(Constraint_vio2,2), 'linestyle', '--', 'linewidth',linewidth);hold on;
le = legend('PProx-PDA $\rho=$ constant','PProx-PDA-IA $\rho = r$');
xl = xlabel('Iteration Number','FontSize',fontsize);
yl = ylabel('Consensus Error','FontSize',fontsize);
%grid on;
set(gca,'FontSize',fontsize);
set(le,'Interpreter','latex');
set(xl,'Interpreter','latex');
set(yl,'Interpreter','latex');
print(gcf,'ppda_log_reg_cons_vio.eps');
%save(gcf, 'cons_vio_bin_larg_stepsize.fig')
savefig('figs/ppda_log_reg_cons_vio.fig');
print(gcf,'-r300','-depsc','figs/ppda_log_reg_cons_vio.eps');



file_name = 'Perturbed prox_PDA_binary_result.mat';
save(file_name);


