% distributed SPCA: min -x'D'Dx +\lambda \|x\|_1 s.t \|x\|_2 <=1
clc;
clear all

close all;

Radius = .5;
max_trial = 1;
max_iter = 500;

fc = @(D,x,xi) -x'* D'* D * x+xi; % functional value
gc = @(D,x) -2 * D'* D * x;  % gradient
fdm_gc = @(D,x,delta) -D' * D * ((x.^2 - (x-delta).^2)./delta);
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
    x_temp1 =reshape(x_temp(:,1),[n,N]);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%% inexact Prog-GPDA with constant stepsize %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Parameters

    lam = 1e-4; % sparsity parameter
    eps = 1e-4; % epsilon error
    gam = .2*eps; % inexact parameter
    rho =  .05/eps;%.05/eps;%.001/gam;%.03/eps; % penalty parameter
    m = 20;
    delta = 10^-4;%1/sqrt(max_iter)/n;
    x = x_temp;
    mu = zeros((edge_index-1)*n,max_iter); % dual variable
    I = eye(n);
    for iter  = 2 : max_iter
        gradient_matrix = zeros(n,N);
        gradient_matrix1 = zeros(n,N);
        for ii = 1 : N
            
            gradient_matrix1(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),x((ii-1)*n+1:ii*n,iter-1));
            for kk=1:n
                g0 = 0;
            for jj=1:m
                phi = unifrnd(0.7,1);%rand;%normrnd(0,1);%normrnd(-0.01,0.01);
                xi = normrnd(0,0.01);
                xi=.001*xi;
                %delta = 1e-4;
                ekk = I(:,kk);
                f1 = fc(D((ii-1)*bs+1:ii*bs,:), x((ii-1)*n+1:ii*n,iter-1)+delta*phi*ekk,xi);
                xi = normrnd(0,0.01);
                xi=.001*xi;
                f2 = fc(D((ii-1)*bs+1:ii*bs,:), x((ii-1)*n+1:ii*n,iter-1),xi);
                g1=(f1-f2)./(delta*phi);
                %f1 = fdm_gc(D((ii-1)*bs+1:ii*bs,:), x((ii-1)*n+1:ii*n,iter-1), delta);
                g0 = g0 + g1;
            end
            gradient_matrix(kk,ii) = g0/m;
            end
            
        end
        norm(gradient_matrix-gradient_matrix1,inf)
        
        gradient = reshape(gradient_matrix,[N*n,1]); % stack the gradients
        d = rho*(A'*A)*x(:,iter-1)+(1-rho*gam)*A'*mu(:,iter-1) + gradient;
        % update x
        u = x(:,iter-1)-d/(rho * l_max); % u stacks all x's
        x_matrix = reshape(u,[n,N]);
        
        
        
        for ii=1:N % for d1 variables apply sof-threshhold
            x_matrix(:,ii) = soft(x_matrix(:,ii),lam/(rho*l_max));
        end
        
        for ii=1:N % for d2 varibale project onto the l2 ball
            if norm(x_matrix(:,ii))>1
                x_matrix(:,ii) = x_matrix(:,ii)/norm(x_matrix(:,ii));
            end
        end
        % for N-(d1+d1) project into positive orthant
%         for ii=d1+d2+1:N
%             x_copy = x_matrix(:,ii);
%             x_copy(find(x_copy<0)) = -x_copy(find(x_copy<0));
%             x_matrix(:,ii) = x_copy;
%         end
        
        x(:,iter) = reshape(x_matrix,[N*n,1]); % stack all x's
        %update dual
        mu(:,iter) = (1-rho*gam)*mu(:,iter-1) + rho * A * x(:,iter);
        x_avg11 = sum(x_matrix,2)/N;
        
        % calculate the opt-gap
        delta = 1e-4;
        full_grad = gc(D,x_avg11)+ones(n,1)*lam/N;
        w = x_avg11 - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w = w/norm(w);
        end
%         w(find(w<0)) = -w(find(w<0));% project into positive orthant
        Opt_gap11(iter-1,trial) = norm(x_avg11-w)^2;
        Constraint_vio11(iter-1,trial) =  norm(A*x(:,iter))^2;
    end
 
   
   
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%% distributed subgradient (DSG) rh0 = 1/r %%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     x_dsg_mat = rand(n,N);
%     v = x_dsg_mat;
%     %u = zeros(n,N);
%     PW = zeros(N,N); % Metropolic-weight matrix
%     W = zeros(N,N);
%     for ii = 1 : N
%         for jj = ii+1 : N
%             if Adj(ii,jj) == 1
%                 PW(ii,jj) = 1/(1+max(degree(ii), degree(jj)));
%                 PW(jj,ii) = PW(ii,jj);
%             end
%         end
%         PW(ii,ii) = 1-sum(PW(ii,:));
%     end
%     
%     for ii = 1 : N
%         for jj = 1 : N
%             W(ii,jj)=1/N;
%         end
%     end
%     
%     for iter  = 2 : max_iter
%         alpha = .01*sqrt(log(trial+1))/(iter);
%         grad_dsg = zeros(n,N);
%         
%         % take the average
%         %v = x_dsg_mat*W;
%         for ii=1:N
%             v(:,ii) = 0;
%             for jj = 1 : N
%                 v(:,ii) = v(:,ii) +  W(ii,jj) * x_dsg_mat(:,jj);
%             end
%             % calculate the gradient in the average point
%             grad_dsg(:,ii) = gc( D((ii-1)*bs+1:ii*bs,:), v(:,ii) );
%         end
%         
%         % update x
%         for ii=1:N
%             x_dsg_mat(:,ii) = v(:,ii) - alpha*grad_dsg(:,ii);
%         end
%         
%         
%         % projection
%         for ii=1:d1 % for d1 variable project onto l1
%             x_dsg_mat(:,ii) = soft(x_dsg_mat(:,ii),N*lam*alpha);
%         end
%         for ii=d1+1:d1+d2 % for d2 varibale project onto the l2 ball
%             if norm(x_dsg_mat(:,ii))>1
%                 x_dsg_mat(:,ii) = x_dsg_mat(:,ii)/norm(x_dsg_mat(:,ii));
%             end
%         end
%         % project into positive orthant
%         for ii=d1+d2+1:N % for N-d1-d2 varibale project onto positive orthant
%             u_copy = x_dsg_mat(:,ii);
%             u_copy(find(x_dsg_mat(:,ii)<0)) = -u_copy(find(x_dsg_mat(:,ii)<0));
%             x_dsg_mat(:,ii) = u_copy;
%         end
%         
%         x_avg_dsg = sum(x_dsg_mat,2)/N;
%         full_grad = gc(D,x_avg_dsg)+ones(n,1)*lam/N;
%         % calculating the opt-gap
%         w = x_avg_dsg - full_grad;
%         w = soft(w, lam); % project into l1
%         if norm(w)>1
%             w=w/norm(w);
%         end
%                 w(find(w<0)) = -w(find(w<0));% project into positive orthant
% 
%         Opt_gap_dsg(iter-1,trial) = norm(x_avg_dsg-w)^2;
%         x_dsg_vec = reshape(x_dsg_mat,[N*n,1]);
%         Constraint_vio_dsg(iter-1,trial) =  norm(A*x_dsg_vec)^2;
%     end

%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%% distributed subgradient (DSG) rh0 = 1/sqrt(r) %%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     x_dsgsq_mat = randn(n,N);
%     v = x_dsgsq_mat;
%     u = zeros(n,N);
%     PW = zeros(N,N); % Metropolic-weight matrix
%     W = zeros(N,N);
%     for ii = 1 : N
%         for jj = ii+1 : N
%             if Adj(ii,jj) == 1
%                 PW(ii,jj) = 1/(1+max(degree(ii), degree(jj)));
%                 PW(jj,ii) = PW(ii,jj);
%             end
%         end
%         PW(ii,ii) = 1-sum(PW(ii,:));
%     end
%     
%     for ii = 1 : N
%         for jj = 1 : N
%             W(ii,jj)=1/N;
%         end
%     end
%     
%     for iter  = 2 : max_iter
%         alpha = .005*log(trial+1)/sqrt(iter);
%         grad_dsgsq = zeros(n,N);
%         
%         % take the average
%         %v = x_dsgsq_mat*W;
%         for ii=1:N
%             v(:,ii) = 0;
%             for jj = 1 : N
%                 v(:,ii) = v(:,ii) +  W(ii,jj) * x_dsgsq_mat(:,jj);
%             end
%             % calculate the gradient in the average point
%             grad_dsgsq(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),v(:,ii));
%         end
%         
%         % update x
%         for ii=1:N
%             x_dsgsq_mat(:,ii) = v(:,ii) - alpha*grad_dsgsq(:,ii);
%         end
%         %     projection
%         for ii=1:d1 % for d1 variable project onto l1
%             x_dsgsq_mat(:,ii) = soft(x_dsgsq_mat(:,ii),N*lam*alpha);
%         end
%         for ii=d1+1:d1+d2 % for d2 varibale project onto the l2 ball
%             if norm(x_dsgsq_mat(:,ii))>1
%                 x_dsgsq_mat(:,ii) = x_dsgsq_mat(:,ii)/norm(x_dsgsq_mat(:,ii));
%             end
%         end
%         % project into positive orthant
%         for ii=d1+d2+1:N % for N-d1-d2 varibale project onto positive orthant
%             u_copy = x_dsgsq_mat(:,ii);
%             u_copy(find(x_dsgsq_mat(:,ii)<0)) = -u_copy(find(x_dsgsq_mat(:,ii)<0));
%             x_dgs_mat(:,ii) = u_copy;
%         end
%         
%         x_avg_dsgsq = sum(x_dsgsq_mat,2)/N;
%         full_grad = gc(D,x_avg_dsgsq)+ones(n,1)*lam/N;
%         % calculating the opt-gap
%         w = x_avg_dsgsq - full_grad;
%         w = soft(w, lam); % project into l1
%         if norm(w)>1
%             w=w/norm(w);
%         end
%                 w(find(w<0)) = -w(find(w<0));% project into positive orthant
% 
%         Opt_gap_dsgsq(iter-1,trial) = norm(x_avg_dsgsq-w)^2;
%         x_dsgsq_vec = reshape(x_dsgsq_mat,[N*n,1]);
%         Constraint_vio_dsgsq(iter-1,trial) =  norm(A*x_dsgsq_vec)^2;
%     end
%    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%% RGF %%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     %% Randomized Gradient-free (RGF) Deming Yuan et al IEEE Transaction on neural networks and learning sustems
% 
    x_rgf_mat = x_temp1;%rand(n,N);
    v = x_rgf_mat;
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
        gamma = .01*sqrt(log(trial+1))/(iter);
        %gamma = 5/sqrt(iter);
        grad_rgf = zeros(n,N);
        
        % take the average
        %v = x_rgf_mat*W;
        for ii=1:N
            v(:,ii) = 0;
            for jj = 1 : N
                v(:,ii) = v(:,ii) +  1/N * x_rgf_mat(:,jj);
            end
            % calculate the gradient in the average point with FDM
            
              for kk=1:n
                g0 = 0;
            for jj=1:m
                phi = unifrnd(0.5,1);%rand;%normrnd(0,1);%normrnd(-0.01,0.01);
                xi = normrnd(0,0.01);
                xi=.01*xi;
                %delta = 1e-4;
                ekk = I(:,kk);
                f1 = fc(D((ii-1)*bs+1:ii*bs,:), v(:,ii)+delta*phi*ekk,xi);
                xi = normrnd(0,0.01);
                xi=.01*xi;
                f2 = fc(D((ii-1)*bs+1:ii*bs,:), v(:,ii),xi);
                g1=(f1-f2)./(delta*phi);
                %f1 = fdm_gc(D((ii-1)*bs+1:ii*bs,:), x((ii-1)*n+1:ii*n,iter-1), delta);
                g0 = g0 + g1;
            end
            grad_rgf(kk,ii) = g0/m;
              end
            
%             m = 10;
%             g0 = zeros(n,1);
%             for jj=1:m
%                 phi = normrnd(0,1);
%                 mu = 1e-5;
%                 xi = normrnd(-0.01,0.01);
%                 f1 = [];
%                 f1 = fdm_gc(D((ii-1)*bs+1:ii*bs,:), v(:,ii), (mu+xi)*phi);
%                 g0 = g0 + f1*phi*phi;
%             end
%             grad_rgf(:,ii) = g0/m;
         end
      
        % update x
        for ii=1:N
            x_rgf_mat(:,ii) = v(:,ii) - gamma*grad_rgf(:,ii);
        end
        
        % projection
        for ii=1:N % for d1 variable project onto l1
            x_rgf_mat(:,ii) = soft(x_rgf_mat(:,ii),N*lam*gamma);
        end
        for ii=1:N % for d2 varibale project onto the l2 ball
            if norm(x_rgf_mat(:,ii))>1
                x_rgf_mat(:,ii) = x_rgf_mat(:,ii)/norm(x_rgf_mat(:,ii));
            end
        end
        % project into positive orthant
%         for ii=d1+d2+1:N % for N-d1-d2 varibale project onto positive orthant
%             u_copy = x_rgf_mat(:,ii);
%             u_copy(find(x_rgf_mat(:,ii)<0)) = -u_copy(find(x_rgf_mat(:,ii)<0));
%             x_rgf_mat(:,ii) = u_copy;
%         end
        
        
        x_avg_rgf = sum(x_rgf_mat,2)/N;
        full_grad = gc(D,x_avg_rgf)+ones(n,1)*lam/N;
        % calculating the opt-gap
        w = x_avg_rgf - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w=w/norm(w);
        end
        %        w(find(w<0)) = -w(find(w<0));% project into positive orthant

        Opt_gap_rgf(iter-1,trial) = norm(x_avg_rgf-w)^2;
        x_rgf_vec = reshape(x_rgf_mat,[N*n,1]);
        Constraint_vio_rgf(iter-1,trial) =  norm(A*x_rgf_vec)^2;
    end
    
    
    
  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%% SGD %%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%    
    x_sgd_mat = x_temp1;%rand(n,N);
    v = x_sgd_mat;
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
        %gamma = 0.01/N/n/sqrt(iter); %% similar to learning rate in machine learning
        gamma = 5/sqrt(iter);
        grad_sgd = zeros(n,N);
        
        % take the average
        %v = x_sgd_mat*W;
        for ii=1:N
            v(:,ii) = 0;
            for jj = 1 : N
                v(:,ii) = v(:,ii) +  1/N * x_sgd_mat(:,jj);
            end
            % calculate the gradient in the average point with FDM
            
            
             for kk=1:n
                g0 = 0;
            for jj=1:m
                phi = unifrnd(0.5,1);%rand;%normrnd(0,1);%normrnd(-0.01,0.01);
                xi = normrnd(0,0.01);
                xi=.01*xi;
                %delta = 1e-4;
                ekk = I(:,kk);
                f1 = fc(D((ii-1)*bs+1:ii*bs,:), v(:,ii)+delta*phi*ekk,xi);
                xi = normrnd(0,0.01);
                xi=.01*xi;
                f2 = fc(D((ii-1)*bs+1:ii*bs,:), v(:,ii),xi);
                g1=(f1-f2)./(delta*phi);
                %f1 = fdm_gc(D((ii-1)*bs+1:ii*bs,:), x((ii-1)*n+1:ii*n,iter-1), delta);
                g0 = g0 + g1;
            end
            grad_sgd(kk,ii) = g0/m;
             end
              for ii = 1 : N
        %    grad_sgd(:,ii) = gc(D((ii-1)*bs+1:ii*bs,:),v(:,ii));
              end
%             m = 2;
%             g0 = zeros(n,1);
%             for jj=1:m
%                 mu = 1e-5;
%                 xi = normrnd(-0.01,0.01);
%                 f1 = [];
%                 r = randi(N, 1);
%                 f1 = fdm_gc(D((r-1)*bs+1:r*bs,:), v(:,ii), mu+xi);
%                 g0 = g0 + f1;
%             end
%             grad_sgd(:,ii) = g0/m;
         end
      
        % update x
        for ii=1:N
            x_sgd_mat(:,ii) = v(:,ii) - gamma*grad_sgd(:,ii);
        end
        
        % projection
        for ii=1:N % for d1 variable project onto l1
            x_sgd_mat(:,ii) = soft(x_sgd_mat(:,ii),N*lam*gamma);
        end
        for ii=1:N % for d2 varibale project onto the l2 ball
            if norm(x_sgd_mat(:,ii))>1
                x_sgd_mat(:,ii) = x_sgd_mat(:,ii)/norm(x_sgd_mat(:,ii));
            end
        end
        % project into positive orthant
%         for ii=d1+d2+1:N % for N-d1-d2 varibale project onto positive orthant
%             u_copy = x_sgd_mat(:,ii);
%             u_copy(find(x_sgd_mat(:,ii)<0)) = -u_copy(find(x_sgd_mat(:,ii)<0));
%             x_sgd_mat(:,ii) = u_copy;
%         end
        
        
        x_avg_sgd = sum(x_sgd_mat,2)/N;
        full_grad = gc(D,x_avg_sgd)+ones(n,1)*lam/N;
        % calculating the opt-gap
        w = x_avg_sgd - full_grad;
        w = soft(w, lam); % project into l1
        if norm(w)>1
            w=w/norm(w);
        end
             %   w(find(w<0)) = -w(find(w<0));% project into positive orthant

        Opt_gap_sgd(iter-1,trial) = norm(x_avg_sgd-w)^2;
        x_sgd_vec = reshape(x_sgd_mat,[N*n,1]);
        Constraint_vio_sgd(iter-1,trial) =  norm(A*x_sgd_vec)^2;
    end
    
end 
    

%% plot the results
linewidth = 2.5;
fontsize = 14;
MarkerSize = 10;
m = 0.1;

figure(5);
% plot all trials
for trial = 1:max_trial
    semilogy(Opt_gap11(:,trial),'--','color',[1,0.7,0.7-m]);hold on;
   % semilogy(Opt_gap_dsg(:,trial),'--', 'color',[.5,0.7,1-m]);hold on;
   % semilogy(Opt_gap_dsgsq(:,trial),'--', 'color',[.4,0.8,1-m]);hold on;
    semilogy(Opt_gap_rgf(:,trial),'--', 'color',[.8,1,0.5-m]);hold on;
    semilogy(Opt_gap_sgd(:,trial),'--', 'color',[0 0 1-m]);hold on;
end
 
%plot the average
h1 = semilogy(mean(Opt_gap11,2),'linewidth',linewidth, 'color', [1-m,0,0]);hold on;
%h2= semilogy(mean(Opt_gap_dsg,2),'linewidth',linewidth, 'color', [0,0,1-m]);hold on;
%h3 = semilogy(mean(Opt_gap_dsgsq,2),'linewidth',linewidth, 'color', [0,0,0.8-m]);hold on;
h4 = semilogy(mean(Opt_gap_rgf,2),'linewidth',linewidth, 'color', [0,1-m,0]);hold on;
h5 = semilogy(mean(Opt_gap_sgd,2),'linewidth',linewidth, 'color', [0 0 1-m]);hold on;


%le = legend([h1,h2,h3,h4, h5],'PProx-PDA constant $\rho$','DSG $\rho =\mathcal{O} (1/r)$', 'DSG $\rho =\mathcal{O} (1/\sqrt{r})$','RGF', 'Steepest Grad');
le = legend([h1,h4,h5],'PZO-PDA','RGF','ZO-SGD');
xl = xlabel('Iteration','FontSize',fontsize);
yl = ylabel('Optimal Residual','FontSize',fontsize);
set(le,'Interpreter','latex');
set(xl,'Interpreter','latex');
set(yl,'Interpreter','latex');
 print(gcf,'-r300','-depsc','/home/ehsankf/research/ZerothOrder/Codes-Haj/codes/figs/opt_gap_3part_next.eps')
%print(gcf,'-r300','-depsc','figs/opt_gap_3part_next.eps');
%surf(peaks)
%savefig('figs/opt_gap_3part_next.fig');

figure(6);
% plot all trials
for trial = 1:max_trial
    semilogy(Constraint_vio11(:,trial),'--','color',[1,0.7,0.7-m]);hold on;
    %semilogy(Constraint_vio_dsg(:,trial),'--', 'color',[.5,0.7,1-m]);hold on;
    %semilogy(Constraint_vio_dsgsq(:,trial),'--', 'color',[.4,0.8,1-m]);hold on;
    semilogy(Constraint_vio_rgf(:,trial),'--', 'color',[.8,1,0.5-m]);hold on;
    semilogy(Constraint_vio_sgd(:,trial),'--', 'color',[0 0 1-m]);hold on;
end
%plot the average
p1 = semilogy(mean(Constraint_vio11,2), 'linewidth',linewidth, 'color', [1-m,0,0]);hold on;
%p2 = semilogy(mean(Constraint_vio_dsg,2), 'linewidth',linewidth, 'color', [0,0,1-m]);hold on;
%p3 = semilogy(mean(Constraint_vio_dsgsq,2), 'linewidth',linewidth, 'color', [0,0,0.8-m]);hold on;
p4 = semilogy(mean(Constraint_vio_rgf,2), 'linewidth',linewidth, 'color', [0,1-m,0]);hold on;
p5 = semilogy(mean(Constraint_vio_sgd,2), 'linewidth',linewidth, 'color', [0 0 1-m]);


%le = legend([p1,p2,p3,p4,p5],'PProx-PDA constant $\rho$','DSG $\rho =\mathcal{O} (1/r)$', 'DSG $\rho =\mathcal{O} (1/\sqrt{r})$','RGF', 'Steepest Grad');
le = legend([p1,p4,p5],'PZO-PDA','RGF','ZO-SGD');
xl = xlabel('Iteration','FontSize',fontsize);
yl = ylabel('Constraint Violation','FontSize',fontsize);
set(le,'Interpreter','latex');
set(xl,'Interpreter','latex');
set(yl,'Interpreter','latex');
%print(gcf,'-r300','-depsc','figs/cons_vio_3part_next.eps');
%savefig('figs/cons_vio_3part_next.fig');
print(gcf,'-r300','-depsc','/home/ehsankf/research/ZerothOrder/Codes-Haj/codes/figs/cons_vio_3part_next.eps')

% file_name = 'DSPCA_all_trials.mat';
% save(file_name);



