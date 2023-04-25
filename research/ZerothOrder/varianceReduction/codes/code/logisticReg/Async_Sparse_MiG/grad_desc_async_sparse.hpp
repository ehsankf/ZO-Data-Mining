#ifndef GRADDESCASYNCSPARSE_H
#define GRADDESCASYNCSPARSE_H
#include "blackbox.hpp"
#include <iostream>
#include <atomic>

namespace grad_desc_async_sparse {

    void Partial_Gradient(double* full_grad_core, size_t thread_no, double* X
        , double* Y, size_t* Jc, size_t* Ir, std::atomic<double>* full_grad
        , size_t N, blackbox* model, size_t _thread, double* _weights = NULL
        , std::atomic<double>* reweight_diag = NULL);
    double* Comp_Full_Grad_Parallel(double* full_grad_core, size_t thread_no
        , double* X, double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model
        , double* _weights = NULL, std::atomic<double>* reweight_diag = NULL);

    // KroMagnon
    std::vector<double>* KroMagnon_Async(double* X, double* Y, size_t* Jc, size_t* Ir
        , size_t N, blackbox* model, size_t iteration_no, size_t thread_no
        , double L, double step_size, bool is_store_result);
    void KroMagnon_Async_Inner_Loop(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
        , std::atomic<double>* x, blackbox* model, size_t m, size_t inner_iters
        , double step_size, std::atomic<double>* reweight_diag, double* full_grad_core
        , double* full_grad);

    // ASAGA
    std::vector<double>* ASAGA_Async(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
        , blackbox* model, size_t iteration_no, size_t thread_no, double L
        , double step_size, bool is_store_result = false);
    void ASAGA_Async_Loop(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
        , std::atomic<double>* x, blackbox* model, size_t inner_iters, double step_size
        , double* reweight_diag, std::atomic<double>* grad_core_table, std::atomic<double>* aver_grad
        , std::vector<double>* losses, bool is_store_result);

    // MiG
    void MiG_Inner_Loop(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
        , std::atomic<double>* x, std::atomic<double>* aver_x, blackbox* model
        , size_t m, size_t inner_iters, double step_size, double theta
        , std::atomic<double>* reweight_diag, double* full_grad_core, double* full_grad
        , double* x_tilda);
    std::vector<double>* MiG_Async(double* X, double* Y, size_t* Jc, size_t* Ir, size_t N
        , blackbox* model, size_t iteration_no, size_t thread_no, double L
        , double sigma, double step_size, bool is_store_result);
}

#endif
