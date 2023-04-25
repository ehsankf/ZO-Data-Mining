#include <math.h>
#include "mex.h"
#include <string.h>
#include "logistic_functions.h"
#include "utilities.h"
#include <random>
#include <iostream>

using namespace std;
/*
	USAGE:
	hist = SVRG(w, Xt, y, lambda, alpha, iters_outer);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; updated in place
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
	lambda - scalar regularization param
	alpha -  learning rate or step-size
	iters_outer - number of iterations
	==================================================================
	OUTPUT PARAMETERS:
	hist = array of function values after each outer loop.
		   Computed ONLY if explicitely asked for output in MATALB.
*/

/// SVRG runs the SVRG algorithm for solving regularized 
/// logistic regression on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments

mxArray* SVRG(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, lambda1, Lmax, iters_outer, alpha;	

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d, minibatch_size; // Dimensions of problem
	long iters_inner; // Number of inner loops
	bool evalf = false; // set to true if function values should be evaluated
	double *mu, *mu1;
	double *g_prev;
	double *g_tilda; 
        double **Matrix;
	double *w_prev;
        //double *ww_prev;
	double *w_tilda;
	double *hist; // Used to store function value at points in history
        double tdiff, max, stdiff;

	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w  = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y  = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
        lambda1 = mxGetScalar(prhs[4]); // Regularization parameter
	alpha  = mxGetScalar(prhs[5]); // Lmax (constant)
	iters_outer = mxGetScalar(prhs[6]); // outer loops (constant)
        minibatch_size = mxGetScalar(prhs[7]); //


	if (nlhs == 1) {
		evalf = true;
	}
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
        int num_batches_train = (int) n / minibatch_size;
        printf("num_batches_train %d", num_batches_train);
	iters_inner = 2 * n; // Number of inner iterations
	
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_batches_train-1);

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	mu = new double[d];
        mu1 = new double[d];
	g_prev  = new double[d];
	g_tilda = new double[d];
	w_prev  = new double[d];
        //ww_prev = new double[d];
	w_tilda = new double[d];
        Matrix = new double*[num_batches_train];
        for (i = 0; i < num_batches_train; i++)
           Matrix[i] = new double[d];
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters_outer + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

         // Initiate matrix
        for (i = 0; i < num_batches_train; i++)
	for (j = 0; j < d; j++) 
            Matrix[i][j] = 0;

	// Initiate w
	for (j = 0; j < d; j++) {
		w_prev[j] = 0;
		w_tilda[j] = w_prev[j];
	}

        //compute_zeroth_full_gradient(Xt, w_tilda, y, mu, n, d, lambda);
        printf("Hii");

	// Outer loop
	for (k = 0; k < iters_outer; k++) {

                printf("Itr %d\n", k);
        
		// Evaluate function value if output requested
		if (evalf == true) {		
			hist[k] = compute_function_value(w_tilda, Xt, y, n, d, lambda, lambda1);			
		}

		// mu with the full gradient 
		//compute_full_gradient(Xt, w_tilda, y, mu1, n, d, lambda);
                compute_zeroth_full_gradient(Xt, w_tilda, y, mu, n, d, lambda, lambda1, k+1);
                /*max = std::abs(mu1[0] - mu[0]);
                for (j = 0; j < d; j++){
		       tdiff = std::abs(mu1[j] - mu[j]);
                       if(tdiff > max){
                            max = tdiff;
                            stdiff = mu1[j] - mu[j];}
		}
                printf("max: %f stdiff %f", max, stdiff);*/
		/*for (j = 0; j < d; j ++){
			//mu[j] += lambda * w_tilda[j];
            ww_prev[j] = 0;
		}*/

		// Inner loop
		for (i = 0; i < iters_inner; i++) {
                         
                        int train_index = dis(gen);//rand() % num_batches_train;
			int left = train_index * minibatch_size;
			int right = (train_index + 1) * minibatch_size;
                        
			if (train_index == num_batches_train-1) 
					right = n;
                        int size = right - left;
			//long idx = dis(gen);
            
			// Compute gradient of last inner iter
			//compute_partial_gradient(Xt, w_prev, y, g_prev, n, d, lambda, idx);
			compute_zeroth_partial_gradient(Xt, w_prev, y, g_prev, n, d, lambda, lambda1, left, right, 1);

			// Compute the gradient of last outer iter
			//compute_partial_gradient(Xt, w_tilda, y, g_tilda, n, d, lambda, idx);
                        compute_zeroth_partial_gradient(Xt, w_tilda, y, g_tilda, n, d, lambda, lambda1, left, right, 1);

                        for (j = 0; j < d; j ++){
				g_prev[j] =  g_prev[j] / size;
                                g_tilda[j] =  g_tilda[j] / size;
			}

			// add gradient of l2 reg
			for (j = 0; j < d; j ++){
				g_prev[j] += lambda * w_prev[j];
				//g_tilda[j] += lambda * w_tilda[j]; 
                                 if(w_prev[j] >= 0)
                                     g_prev[j] += lambda1;      
                                else
                                     g_prev[j] -= lambda1;
			}

                        
                        
			// Update the test point 
			update_test_point_SVRG(w_prev, g_prev, g_tilda, mu, alpha, d);

                        /*double param = alpha * 1e-6;
                        for (j = 0; j < d; j ++){
                          if(w_prev[j] > param)
                             w_prev[j] -= param;
                          else if(w_prev[j] < -param)
                             w_prev[j] += param;
                          else
                             w_prev[j] = 0;
                        }*/
            
			/*for (j = 0; j < d; j ++){
				ww_prev[j] += w_prev[j];
			}*/
		}

		for (j = 0; j < d; j ++) {
            w_tilda[j] = w_prev[j];
            //w_tilda[j] = ww_prev[j]/iters_inner;
            //w_prev[j] = w_tilda[j];
		}
    }
    if (evalf == true) {
        hist[(int)(iters_outer)] = compute_function_value(w_tilda, Xt, y, n, d, lambda, lambda1); //new
    }
    
	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] mu;
	delete[] g_prev;
	delete[] g_tilda;
	delete[] w_prev;
	delete[] w_tilda;
        delete[] Matrix;

	if (evalf == true) { return plhs; }
	else { return 0; }
}

/// Entry function of MATLAB
/// nlhs - number of output parameters
/// *plhs[] - array poiters to the outputs
/// nrhs - number of input parameters
/// *prhs[] - array of pointers to inputs
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// First determine, whether the data matrix is stored in sparse format.
	// If it is, use more efficient algorithm
	if (mxIsSparse(prhs[1])) {
		plhs[0] = SVRG(nlhs, prhs);
	}
	else {
		plhs[0] = SVRG(nlhs, prhs);
	}
}
