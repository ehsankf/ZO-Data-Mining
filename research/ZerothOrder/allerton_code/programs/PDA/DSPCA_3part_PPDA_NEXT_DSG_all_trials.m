% distributed SPCA: min -x'D'Dx +\lambda \|x\|_1 s.t \|x\|_2 <=1
% Davood, Mar 15 2017
clc;
clear;
close all;

Radius = .5;
max_trial = 20;
max_iter = 500;

fc = @(D,x) -x'* D'* D * x; % functional value
gc = @(D,x) -2 * D'* D * x;  % gradient
soft = @(z,lam) sign(z).*max(abs(z)-lam,0); % soft-threshhold


Opt_gap11 = zeros(max_iter-1,max_trial);
Opt_gap12 = zeros(max_iter-1,max_trial);
Opt_gap_dsg = zeros(max_iter-1,max_trial);
Opt_gap_next = zeros(max_iter-1,max_trial);

Constraint_vio11 = zeros(max_iter-1,max_trial);
Constraint_vio12 = zeros(max_iter-1,max_trial);
Constraint_vio_dsg = zeros(max_iter-1,max_trial);
Constraint_vio_next = zeros(max_iter-1,max_trial);

N = 10;  % # of agents
bs = 100; % batch size
K = bs * N; % number of data points
n = 10; % problem dimension (number of variables)
seed = 1;  % for random number generators

d1 = floor(N/3);
d2 = floor(N/3);
d3 = N-(d1+d2);

x_matrix = zeros(n,N);

for trial = 1 : max_trial
%% data
D = rand(K,n); % K * n

%% Network Topology Generation
while (1)
    [error,Adj, degree, xy, Lmatrix]=randomgraph(N,Radius,seed);
    if ~error
        break;
    end
end
D_tilde = diag(degree); %% diagonal degree matrix
num_of_e = sum(Adj(:))/2;
A = zeros(num_of_e,N);
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
E = eye(N,N);
A = kron(A,eye(n));
B = abs(A);
Lm = A' * A;
l_max = eigs(Lm,1);

    trial
    x_temp = rand(N*n,max_iter)*log(trial);
    x_temp(:,1) = randn(N*n,1);
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% inexact Prog-GPDA with constant stepsize %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Parameters

            lam = 10; % sparsity parameter
            eps = 1e-4; % epsilon error
            gam = .2*eps; % inexact parameter
            rho =  .3/eps; % penalty parameter

    x = x_temp;
    mu = zeros((edge_index-1)*n,max_iter); % dual variable
    
    for iter  = 2 : max_iter
        gradient_matrix = zeros(n,N);
        for ii = 1 : N
            gradient_matrix(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),x((ii-1)*n+1:ii*n,iter-1));
        end
        
        gradient = reshape(gradient_matrix,[N*n,1]); % stack the gradients
        d = rho*(A'*A)*x(:,iter-1)+(1-rho*gam)*A'*mu(:,iter-1) + gradient;
        % update x
        u = x(:,iter-1)-d/(rho * l_max); % u stacks all x's
        x_matrix = reshape(u,[n,N]);
        for ii=1:d1 % for d1 variables apply sof-threshhold
            x_matrix(:,ii) = soft(x_matrix(:,ii),lam/(N*rho*l_max));
        end
        
        for ii=d1+1:d1+d2 % for d2 varibale project onto the l2 ball
            if norm(x_matrix(:,ii))>1
                x_matrix(:,ii) = x_matrix(:,ii)/norm(x_matrix(:,ii));
            end
        end
        % for N-(d1+d1) project into positive orthant
        for ii=d1+d2+1:N
            x_copy = x_matrix(:,ii);
            x_copy(find(x_copy<0)) = -x_copy(find(x_copy<0));
            x_matrix(:,ii) = x_copy;
        end
        
        x(:,iter) = reshape(x_matrix,[N*n,1]); % stack all x's
        %update dual
        mu(:,iter) = (1-rho*gam)*mu(:,iter-1) + rho * A * x(:,iter);
        x_avg11 = sum(x_matrix,2)/N;
        
        % calculate the opt-gap
        full_grad = gc(D,x_avg11)+ones(n,1)*lam/N;
        w = x_avg11 - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w = w/norm(w);
        end
        w(find(w<0)) = -w(find(w<0));% project into positive orthant
        Opt_gap11(iter-1,trial) = norm(x_avg11-w)^2;
        Constraint_vio11(iter-1,trial) =  norm(A*x(:,iter))^2;
    end
 
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% inexact Prox-GPDA with increasing accuracy %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    x = x_temp;
    mu = zeros((edge_index-1)*n,max_iter); % dual variable
    
    for iter  = 2 : max_iter
        gam = 1e-4/iter; % inexact parameter
        rho = 30*(iter); 
        
        gradient_matrix = zeros(n,N);
        for ii = 1 : N
            gradient_matrix(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),x((ii-1)*n+1:ii*n,iter-1));
        end
        
        gradient = reshape(gradient_matrix,[N*n,1]); % stack the gradients
        d = rho*(A'*A)*x(:,iter-1)+(1-rho*gam)*A'*mu(:,iter-1) + gradient;
        % update x
        u = x(:,iter-1)-d/(rho * l_max); % u stacks all x's
        x_matrix = reshape(u,[n,N]);
        for ii=1:d1
            x_matrix(:,ii) = soft(x_matrix(:,ii),lam/(N*rho*l_max));
        end
        
        for ii=d1+1:d1+d2 % for d2 varibale project onto the l2 ball
            if norm(x_matrix(:,ii))>1
                x_matrix(:,ii) = x_matrix(:,ii)/norm(x_matrix(:,ii));
            end
        end
        % project into positive orthant
        for ii=d1+d2+1:N
            x_copy = x_matrix(:,ii);
            x_copy(find(x_copy<0)) = -x_copy(find(x_copy<0));
            x_matrix(:,ii) = x_copy;
        end
        
        x(:,iter) = reshape(x_matrix,[N*n,1]); % stack all x's
        %update dual
        mu(:,iter) = (1-rho*gam)*mu(:,iter-1) + rho * A * x(:,iter);
        x_avg12 = sum(x_matrix,2)/N;
        
        % calculate the opt-gap
        full_grad = gc(D,x_avg12)+ones(n,1)*lam/N;
        w = x_avg12 - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w = w/norm(w);
        end
        w(find(w<0)) = -w(find(w<0));% project into positive orthant
        
        Opt_gap12(iter-1,trial) = norm(x_avg12-w)^2;
        Constraint_vio12(iter-1,trial) =  norm(A*x(:,iter))^2;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% distributed subgradient (DSG) rh0 = 1/r %%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_dsg_mat = rand(n,N);
    v = x_dsg_mat;
    %u = zeros(n,N);
    PW = zeros(N,N); % Metropolic-weight matrix
    W = zeros(N,N);
    for ii = 1 : N
        for jj = ii+1 : N
            if Adj(ii,jj) == 1
                PW(ii,jj) = 1/(1+max(degree(ii), degree(jj)));
                PW(jj,ii) = PW(ii,jj);
            end
        end
        PW(ii,ii) = 1-sum(PW(ii,:));
    end
    
    for ii = 1 : N
        for jj = 1 : N
            W(ii,jj)=1/N;
        end
    end
    
    for iter  = 2 : max_iter
        alpha = .01*sqrt(log(trial+1))/(iter);
        grad_dsg = zeros(n,N);
        
        % take the average
        %v = x_dsg_mat*W;
        for ii=1:N
            v(:,ii) = 0;
            for jj = 1 : N
                v(:,ii) = v(:,ii) +  W(ii,jj) * x_dsg_mat(:,jj);
            end
            % calculate the gradient in the average point
            grad_dsg(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),v(:,ii));
        end
        
        % update x
        for ii=1:N
            x_dsg_mat(:,ii) = v(:,ii) - alpha*grad_dsg(:,ii);
        end
        %     projection
        for ii=1:d1 % for d1 variable project onto l1
            x_dsg_mat(:,ii) = soft(x_dsg_mat(:,ii),N*lam*alpha);
        end
        for ii=d1+1:d1+d2 % for d2 varibale project onto the l2 ball
            if norm(x_dsg_mat(:,ii))>1
                x_dsg_mat(:,ii) = x_dsg_mat(:,ii)/norm(x_dsg_mat(:,ii));
            end
        end
        % project into positive orthant
        for ii=d1+d2+1:N % for N-d1-d2 varibale project onto positive orthant
            u_copy = x_dsg_mat(:,ii);
            u_copy(find(x_dsg_mat(:,ii)<0)) = -u_copy(find(x_dsg_mat(:,ii)<0));
            x_dgs_mat(:,ii) = u_copy;
        end
        
        x_avg_dsg = sum(x_dsg_mat,2)/N;
        full_grad = gc(D,x_avg_dsg)+ones(n,1)*lam/N;
        % calculating the opt-gap
        w = x_avg_dsg - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w=w/norm(w);
        end
                w(find(w<0)) = -w(find(w<0));% project into positive orthant

        Opt_gap_dsg(iter-1,trial) = norm(x_avg_dsg-w)^2;
        x_dsg_vec = reshape(x_dsg_mat,[N*n,1]);
        Constraint_vio_dsg(iter-1,trial) =  norm(A*x_dsg_vec)^2;
    end
    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% distributed subgradient (DSG) rh0 = 1/sqrt(r) %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_dsg_mat = randn(n,N);
    v = x_dsg_mat;
    u = zeros(n,N);
    PW = zeros(N,N); % Metropolic-weight matrix
    W = zeros(N,N);
    for ii = 1 : N
        for jj = ii+1 : N
            if Adj(ii,jj) == 1
                PW(ii,jj) = 1/(1+max(degree(ii), degree(jj)));
                PW(jj,ii) = PW(ii,jj);
            end
        end
        PW(ii,ii) = 1-sum(PW(ii,:));
    end
    
    for ii = 1 : N
        for jj = 1 : N
            W(ii,jj)=1/N;
        end
    end
    
    for iter  = 2 : max_iter
        alpha = .005*log(trial+1)/sqrt(iter);
        grad_dsg = zeros(n,N);
        
        % take the average
        %v = x_dsg_mat*W;
        for ii=1:N
            v(:,ii) = 0;
            for jj = 1 : N
                v(:,ii) = v(:,ii) +  W(ii,jj) * x_dsg_mat(:,jj);
            end
            % calculate the gradient in the average point
            grad_dsg(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),v(:,ii));
        end
        
        % update x
        for ii=1:N
            x_dsg_mat(:,ii) = v(:,ii) - alpha*grad_dsg(:,ii);
        end
        %     projection
        for ii=1:d1 % for d1 variable project onto l1
            x_dsg_mat(:,ii) = soft(x_dsg_mat(:,ii),N*lam*alpha);
        end
        for ii=d1+1:d1+d2 % for d2 varibale project onto the l2 ball
            if norm(x_dsg_mat(:,ii))>1
                x_dsg_mat(:,ii) = x_dsg_mat(:,ii)/norm(x_dsg_mat(:,ii));
            end
        end
        % project into positive orthant
        for ii=d1+d2+1:N % for N-d1-d2 varibale project onto positive orthant
            u_copy = x_dsg_mat(:,ii);
            u_copy(find(x_dsg_mat(:,ii)<0)) = -u_copy(find(x_dsg_mat(:,ii)<0));
            x_dgs_mat(:,ii) = u_copy;
        end
        
        x_avg_dsg = sum(x_dsg_mat,2)/N;
        full_grad = gc(D,x_avg_dsg)+ones(n,1)*lam/N;
        % calculating the opt-gap
        w = x_avg_dsg - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w=w/norm(w);
        end
                w(find(w<0)) = -w(find(w<0));% project into positive orthant

        Opt_gap_dsg_sq(iter-1,trial) = norm(x_avg_dsg-w)^2;
        x_dsg_vec = reshape(x_dsg_mat,[N*n,1]);
        Constraint_vio_dsg_sq(iter-1,trial) =  norm(A*x_dsg_vec)^2;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% NEXT alpha = 1/r %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    x_next = rand(n,N);
    x_tilde_next = zeros(n,N);
    y_next = zeros(n,N);
    pi_tilde_next = zeros(n,N);
    grad_next = zeros(n,N);
    for ii=1:N
        grad_next(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),x_next(:,ii));
        y_next(:,ii) = grad_next(:,ii);
        pi_tilde_next(:,ii) = N*y_next(:,ii) - grad_next(:,ii);
    end

    PW = zeros(N,N); % Metropolis-weight matrix
    W = zeros(N,N);
    for ii = 1 : N
        for jj = ii+1 : N
            if Adj(ii,jj) == 1
                PW(ii,jj) = 1/(1+max(degree(ii), degree(jj)));
                PW(jj,ii) = PW(ii,jj);
            end
        end
        PW(ii,ii) = 1-sum(PW(ii,:));
    end
    
    for ii = 1 : N
        for jj = 1 : N
            W(ii,jj)=1/N;
        end
    end
    tau = rand(N,1);
    for iter  = 2 : max_iter
        alpha = (1e-5)*log(trial+1)/(iter);
        %alpha = 0;
        for ii=1:N
            grad_next(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),x_next(:,ii));
            x_tilde_next(:,ii) = (-grad_next(:,ii) + tau(ii)*x_next(:,ii) - pi_tilde_next(:,ii))/tau(ii);
        end
        
        %%%% update x %%%%
        for ii=1:d1 % for d1 variable project onto l1
            x_tilde_next(:,ii) = soft(x_tilde_next(:,ii),lam/tau(ii));
        end
        for ii=d1+1:d1+d2 % for d2 varibale project onto the l2 ball
            if norm(x_tilde_next(:,ii))>1
                 x_tilde_next(:,ii) = x_tilde_next(:,ii)/norm(x_tilde_next(:,ii));
            end
        end
        % project into positive orthant
        for ii=d1+d2+1:N % for N-d1-d2 varibale project onto positive orthant
            u_copy = x_tilde_next(:,ii);
            u_copy(find(x_tilde_next(:,ii)<0)) = -u_copy(find(x_tilde_next(:,ii)<0));
            x_tilde_next(:,ii) = u_copy;
        end
        %%% update z %%%
        z_next = x_next + alpha*(x_tilde_next - x_next);
        %%% consensus step %%%
        x_next = z_next*PW; % take the average
        grad_next_old = grad_next;
        
         for ii=1:N
            grad_next(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),x_next(:,ii));
         end
        y_next = y_next*PW + grad_next - grad_next_old;
        pi_tilde_next = N*y_next-grad_next;
        
        x_avg_next = sum(x_next,2)/N;
        full_grad = gc(D,x_avg_next) + ones(n,1)*lam/N;
        % calculating the opt-gap
        w = x_avg_next - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w=w/norm(w);
        end
        w(find(w<0)) = -w(find(w<0));% project into positive orthant

        Opt_gap_next(iter-1,trial) = norm(x_avg_next-w)^2;
        x_next_vec = reshape(x_next,[N*n,1]);
        Constraint_vio_next(iter-1,trial) =  norm(A*x_next_vec)^2;
    end
end 
    



%x_avg_dsg

%% plot the results
linewidth = 2.5;
fontsize = 14;
MarkerSize = 10;

figure;
% plot all trials
for trial = 1:max_trial
    semilogy(Opt_gap11(:,trial),'--','color',[1,0.7,0.7]);hold on;
    semilogy(Opt_gap12(:,trial),'--', 'color',[.5,0.7,1]);hold on;
    semilogy(Opt_gap_dsg(:,trial),'--', 'color',[.8,1,0.5]);hold on;
    semilogy(Opt_gap_next(:,trial),'--', 'color',[.7,.7,.7]);hold on;
end
 
%plot the average
h1 = semilogy(mean(Opt_gap11,2),'linewidth',linewidth, 'color', 'r');hold on;
h2= semilogy(mean(Opt_gap12,2),'linewidth',linewidth, 'color', 'b');hold on;
h3 = semilogy(mean(Opt_gap_dsg,2),'linewidth',linewidth, 'color', 'g');hold on;
h4 = semilogy(mean(Opt_gap_next,2),'linewidth',linewidth, 'color', 'k');hold on;


le = legend([h1,h2,h3,h4],'PProx-PDA constant $\rho$', 'PProx-PDA-IA $\rho = \mathcal{O}(\sqrt{r})$','DSG $\rho =\mathcal{O} (1/r)$', 'NEXT');
xl = xlabel('Iteration Number','FontSize',fontsize);
yl = ylabel('Stationary Gap','FontSize',fontsize);
set(le,'Interpreter','latex');
set(xl,'Interpreter','latex');
set(yl,'Interpreter','latex');
%print(gcf,'-r300','-depsc','figs/opt_gap_3part_next.eps');
%surf(peaks)
%savefig('figs/opt_gap_3part_next.fig');

figure;
% plot all trials
for trial = 1:max_trial
    semilogy(Constraint_vio11(:,trial),'--','color',[1,0.7,0.7]);hold on;
    semilogy(Constraint_vio12(:,trial),'--', 'color',[.5,0.7,1]);hold on;
    semilogy(Constraint_vio_dsg(:,trial),'--', 'color',[.8,1,0.5]);hold on;
    semilogy(Constraint_vio_next(:,trial),'--', 'color',[.7,0.7,0.7]);hold on;
end
%plot the average
p1 = semilogy(mean(Constraint_vio11,2), 'linewidth',linewidth, 'color', 'r');hold on;
p2 = semilogy(mean(Constraint_vio12,2), 'linewidth',linewidth, 'color', 'b');hold on;
p3 = semilogy(mean(Constraint_vio_dsg,2), 'linewidth',linewidth, 'color', 'g');hold on;
p4 = semilogy(mean(Constraint_vio_next,2), 'linewidth',linewidth, 'color', 'k');


le = legend([p1,p2,p3,p4],'PProx-PDA constant $\rho$', 'PProx-PDA-IA $\rho = \mathcal{O}(\sqrt{r})$','DSG $\rho =\mathcal{O} (1/r)$','NEXT');
xl = xlabel('Iteration Number','FontSize',fontsize);
yl = ylabel('Consensus Error','FontSize',fontsize);
set(le,'Interpreter','latex');
set(xl,'Interpreter','latex');
set(yl,'Interpreter','latex');
%print(gcf,'-r300','-depsc','figs/cons_vio_3part_next.eps');
%savefig('figs/cons_vio_3part_next.fig');

file_name = 'DSPCA_all_trials.mat';
save(file_name);