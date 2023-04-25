function [pred_acc, time1] = Classification(opt_method, epochs)

% Training for multiple-classes: One-vs-Rest

load('..\Data\mnist.mat');

train_y = logical(train_y);
test_y = logical(test_y);

% Training Parameters
N_train_all = size(train_x, 2);
N_test_all  = size(test_x, 2);

N_train = zeros(1,10);
for i=1:10
    N_train(1,i) = sum(train_y(i,:));
end
N_test = zeros(1,10);
for i=1:10
    N_test(1,i) = sum(test_y(i,:));
end

d = size(train_x, 1);

% Tuning parameter  
gamma  = 0.00001;
x      = zeros(d, 10);
max_it = epochs * N_train_all;

% for SGD
d_bound_sgd = 1;  


% Stochastic training
tic
for c = 1:10
    disp(['Training for Class ', num2str(c)]);
    % preparing data
    train_samples = zeros(d, N_train(c));
    train_samples(:,1:N_train(c)) = train_x(:,train_y(c,:));
    for i = 1:10
        if i~=c
            train_samples = [train_samples, train_x(:,train_y(i,:))];
        end
    end
    train_labels(1:N_train(c)) = 1;
    train_labels(N_train(c)+1:N_train_all) = -1;
    
    if strcmp(opt_method, 'FSVRG')
        x(:,c) = Alg_FSVRG(train_samples, train_labels, gamma, max_it);    
    elseif strcmp(opt_method, 'SVRSG')
        x(:,c) = Alg_SVRG(train_samples, train_labels, gamma, max_it);
    else
        x(:,c) = Alg_SSGD(train_samples, train_labels, gamma, max_it, d_bound_sgd);
    end
end
time1 = toc;
disp('Training finished.');


% Testing
err_ct   = 0;
pred_lbl = zeros(N_test_all, 1);
test_ct  = 0;
for c = 1:10
    test_xc = test_x(:,test_y(c,:));
    for i = 1:N_test(c)
        test_ct = test_ct + 1;
        test_s  = test_xc(:, i);
        pred_val = zeros(1,10);
        for p = 1:10
            pred_val(p) = test_s' * x(:,p);
        end
        [pred_val_max, pred_lbl(test_ct)] = max(pred_val);
        if (pred_lbl(test_ct) ~= c)
            err_ct = err_ct + 1;
        end
    end
    clear test_xc;
end

pred_acc = 1 - err_ct/N_test_all;
disp(['Prediction accuracy: ', num2str(pred_acc)]);
end