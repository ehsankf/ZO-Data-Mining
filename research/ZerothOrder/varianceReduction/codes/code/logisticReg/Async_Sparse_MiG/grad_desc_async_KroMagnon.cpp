#include "grad_desc_async_sparse.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <atomic>
#include <random>
#include <cmath>
#include <thread>
#include <mutex>
#include <string.h>

extern size_t MAX_DIM;

void grad_desc_async_sparse::Partial_Gradient(double* full_grad_core, size_t thread_no
    , double* X, double* Y, size_t* Jc, size_t* Ir, std::atomic<double>* full_grad
    , size_t N, blackbox* model, size_t _thread, double* _weights
    , std::atomic<double>* reweight_diag) {
    double* _pf = new double[MAX_DIM];
    double* _prd = new double[MAX_DIM];
    memset(_pf, 0, MAX_DIM * sizeof(double));
    memset(_prd, 0, MAX_DIM * sizeof(double));
    for(size_t i = ceil((double) N / thread_no * (_thread - 1));
            i < ceil((double) N / thread_no * _thread);
            i ++) {
        full_grad_core[i] = model->first_component_oracle_core_sparse(X, Y, Jc, Ir, N, i, _weights);
        for(size_t j = Jc[i]; j < Jc[i + 1]; j ++) {
            _pf[Ir[j]] += X[j] * full_grad_core[i] / (double) N;
            // Compute Re-weight Matrix(Inversed) in First Pass
            if(reweight_diag != NULL)
                _prd[Ir[j]] += 1.0 / (double) N;
        }
    }
    // Atomic Write
    for(size_t i = 0; i < MAX_DIM; i ++) {
        fetch_n_add_atomic(full_grad[i], _pf[i]);
        if(reweight_diag != NULL)
            fetch_n_add_atomic(reweight_diag[i], _prd[i]);
    }
    delete[] _prd;
    delete[] _pf;
}

double* grad_desc_async_sparse::Comp_Full_Grad_Parallel(double* full_grad_core
    , size_t thread_no, double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
    , blackbox* model, double* _weights, std::atomic<double>* reweight_diag) {
    // Thread Pool
    std::vector<std::thread> thread_pool;
    std::atomic<double>* full_grad = new std::atomic<double>[MAX_DIM];
    memset(full_grad, 0, MAX_DIM * sizeof(double));
    for(size_t i = 1; i <= thread_no; i ++) {
        thread_pool.push_back(std::thread(Partial_Gradient, full_grad_core, thread_no
            , X, Y, Jc, Ir, full_grad, N, model, i, _weights, reweight_diag));
    }
    for(auto &t : thread_pool)
        t.join();
    double* full_grad_n = new double[MAX_DIM];
    for(size_t i = 0; i < MAX_DIM; i ++)
        full_grad_n[i] = full_grad[i];
    delete[] full_grad;
    return full_grad_n;
}

std::atomic<int> SVRG_counter(0);
void grad_desc_async_sparse::KroMagnon_Async_Inner_Loop(double* X, double* Y, size_t* Jc
    , size_t* Ir, size_t N, std::atomic<double>* x, blackbox* model, size_t m
    , size_t inner_iters, double step_size, std::atomic<double>* reweight_diag
    , double* full_grad_core, double* full_grad) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    double* inconsis_x = new double[MAX_DIM];
    for(size_t j = 0; j < inner_iters; j ++) {
        int rand_samp = distribution(generator);
        // Inconsistant Read [X].
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++)
            inconsis_x[Ir[k]] = x[Ir[k]];
        SVRG_counter.fetch_add(1);
        double inner_core = model->first_component_oracle_core_sparse(X, Y
                    , Jc, Ir, N, rand_samp, inconsis_x);
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++) {
            size_t index = Ir[k];
            double val = X[k];
            double vr_sub_grad = (inner_core - full_grad_core[rand_samp]) * val
                        + reweight_diag[index] * (full_grad[index] + lambda[0] * inconsis_x[index]) ;
            double incr_x = - step_size * vr_sub_grad;
            // Re-Weighted Sparse Estimate of regularizer
            // regularizer::proximal_operator(regular, incr_x, reweight_diag[index] * step_size
            //      , lambda);
            // Atomic Write
            fetch_n_add_atomic(x[index], incr_x);
        }
    }
    delete[] inconsis_x;
}

std::vector<double>* grad_desc_async_sparse::KroMagnon_Async(double* X
    , double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model, size_t iteration_no
    , size_t thread_no, double L, double step_size, bool is_store_result) {
    std::vector<double>* losses = new std::vector<double>;
    std::atomic<double>* x = new std::atomic<double>[MAX_DIM];
    // "Anticipate" Update Extra parameters
    std::atomic<double>* reweight_diag = new std::atomic<double>[MAX_DIM];
    size_t m = N * 2;
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    copy_vec((double *)x, model->get_model());
    // Init Weight Evaluate
    if(is_store_result) {
        losses->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
    }
    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* outter_x = model->get_model();
        double* full_grad_core = new double[N];
        double* full_grad;
        SVRG_counter = 0;
        // Full Gradient
        if(i == 0) {
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x, reweight_diag);
            // Compute Re-weight Matrix in First Pass
            for(size_t j = 0; j < MAX_DIM; j ++)
                reweight_diag[j] = 1.0 / reweight_diag[j];
        }
        else
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x);

        // Parallel INNER_LOOP
        std::vector<std::thread> thread_pool;
        for(size_t k = 1; k <= thread_no; k ++) {
            size_t inner_iters;
            if(k == 1)
                inner_iters = m - m / thread_no * thread_no + m / thread_no;
            else
                inner_iters = m / thread_no;

            thread_pool.push_back(std::thread(KroMagnon_Async_Inner_Loop, X, Y, Jc, Ir, N
                , x, model, m, inner_iters, step_size, reweight_diag, full_grad_core
                , full_grad));
        }
        for(auto& t : thread_pool)
            t.join();

        model->update_model((double*) x);

        // For Matlab (per m/n passes)
        if(is_store_result) {
            losses->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
        }
        delete[] full_grad_core;
        delete[] full_grad;
    }
    delete[] reweight_diag;
    delete[] x;
    return losses;
}
