#include <iostream>
#include "mex.h"
#include "grad_desc_async_sparse.hpp"
#include "svm.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "least_square.hpp"
#include "utils.hpp"
#include <string.h>

size_t MAX_DIM;
const size_t MAX_PARAM_STR_LEN = 15;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        double *X = mxGetPr(prhs[0]);
        double *Y = mxGetPr(prhs[1]);
        MAX_DIM = mxGetM(prhs[0]);
        size_t N = mxGetN(prhs[0]);
        bool is_sparse = mxGetLogicals(prhs[10])[0];
        size_t* Jc;
        size_t* Ir;
        if(is_sparse) {
            Jc = mxGetJc(prhs[0]);
            Ir = mxGetIr(prhs[0]);
        }
        double *init_weight = mxGetPr(prhs[5]);
        double lambda1 = mxGetScalar(prhs[6]);
        double lambda2 = 0;
        double L = mxGetScalar(prhs[7]);
        double sigma = mxGetScalar(prhs[11]);
        double step_size = mxGetScalar(prhs[8]);
        size_t iteration_no = (size_t) mxGetScalar(prhs[9]);
        bool is_store_result = false;
        if(nlhs >= 1)
            is_store_result = true;

        int regularizer;
        char* _regul = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[4], _regul, MAX_PARAM_STR_LEN);
        if(strcmp(_regul, "L2") == 0) {
            regularizer = regularizer::L2;
        }
        else if(strcmp(_regul, "L1") == 0) {
            regularizer = regularizer::L1;
        }
        else if(strcmp(_regul, "elastic_net") == 0) {
            regularizer = regularizer::ELASTIC_NET;
        }
        else mexErrMsgTxt("400 Unrecognized regularizer.");
        delete[] _regul;

        blackbox* model;
        char* _model = new char[MAX_PARAM_STR_LEN];
        double lambdas[2] = {lambda1, lambda2};
        mxGetString(prhs[3], _model, MAX_PARAM_STR_LEN);
        if(strcmp(_model, "logistic") == 0) {
            model = new logistic(2, lambdas, regularizer);
        }
        else if(strcmp(_model, "least_square") == 0) {
            model = new least_square(2, lambdas, regularizer);
        }
        else if(strcmp(_model, "svm") == 0) {
            model = new svm(2, lambdas, regularizer);
        }
        else mexErrMsgTxt("400 Unrecognized model.");
        //delete[] _model;
        model->set_init_weights(init_weight);

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[2], _algo, MAX_PARAM_STR_LEN);
        double *stored_F;
        std::vector<double>* losses;
        size_t len_stored_F;
        if(strcmp(_algo, "ASAGA") == 0) {
            size_t thread_no = (size_t) mxGetScalar(prhs[12]);
            if(!is_sparse) mexErrMsgTxt("400 Async Methods with Dense Input.");
            losses = grad_desc_async_sparse::ASAGA_Async(X, Y, Jc, Ir, N, model
                , iteration_no, thread_no, L, step_size, is_store_result);
            stored_F = &(*losses)[0];
            len_stored_F = losses->size();
        }
        else if(strcmp(_algo, "KroMagnon") == 0) {
            size_t thread_no = (size_t) mxGetScalar(prhs[12]);
            if(!is_sparse) mexErrMsgTxt("400 Async Methods with Dense Input.");
            losses = grad_desc_async_sparse::KroMagnon_Async(X
                , Y, Jc, Ir, N, model, iteration_no, thread_no, L, step_size, is_store_result);
            stored_F = &(*losses)[0];
            len_stored_F = losses->size();
        }
        else if(strcmp(_algo, "AMiG") == 0) {
            size_t thread_no = (size_t) mxGetScalar(prhs[12]);
            double theta = mxGetScalar(prhs[13]);
            if(!is_sparse) mexErrMsgTxt("400 Async Methods with Dense Input.");
            losses = grad_desc_async_sparse::MiG_Async(X
                , Y, Jc, Ir, N, model, iteration_no, thread_no, L, theta, step_size, is_store_result);
            stored_F = &(*losses)[0];
            len_stored_F = losses->size();
        }
        else mexErrMsgTxt("400 Unrecognized algorithm.");
        delete[] _algo;

        if(is_store_result) {
            plhs[0] = mxCreateDoubleMatrix(len_stored_F, 1, mxREAL);
        	double* res_stored_F = mxGetPr(plhs[0]);
            for(size_t i = 0; i < len_stored_F; i ++)
                res_stored_F[i] = stored_F[i];
        }
        delete[] stored_F;
        delete model;
        delete[] _model;
    } catch(std::string c) {
        std::cerr << c << std::endl;
        //exit(EXIT_FAILURE);
    }
}
