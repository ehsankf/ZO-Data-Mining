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
	hist = Katyusha(w, Xt, y, lambda, Lmax, iters_outer);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; updated in place
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
	lambda - scalar regularization param
	Lmax -  Lipschitz constant
	iters_outer - number of iterations
	==================================================================
	OUTPUT PARAMETERS:
	hist = array of function values after each outer loop.
		   Computed ONLY if explicitely asked for output in MATALB.
*/

/// Katyusha runs the Katyusha algorithm for solving regularized 
/// logistic regression on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments

mxArray* Katyusha(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, Lmax, iters_outer, alpha, tau, tau1, ww;		

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d;   // Dimensions of problem
	long iters_inner; // Number of inner loops
	bool evalf = false; // set to true if function values should be evaluated
	double *mu;
	double *g_prev;
	double *g_tilda;
	double *w_prev;
	double *w_tilda;
    double *x_prev;
    double *x_prev1;
    double *y_prev;
    double *ww_prev;    
	double *hist; // Used to store function value at points in history

	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w  = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y  = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	Lmax = mxGetScalar(prhs[4]);  // Lmax (constant)
	iters_outer = mxGetScalar(prhs[5]); // outer loops (constant)


	if (nlhs == 1) {
		evalf = true;
	}
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters_inner = 2 * n; // Number of inner iterations
    
    Lmax = 3*Lmax;
    tau = min(0.5, sqrt(2*n*lambda/Lmax));
    alpha = 1.0 / (tau*Lmax);
    tau1  = 1 + lambda*alpha;
    
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n-1);

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	mu = new double[d];
	g_prev  = new double[d];
	g_tilda = new double[d];
	w_prev  = new double[d];
	w_tilda = new double[d];
    x_prev  = new double[d]; 
    x_prev1 = new double[d];
    y_prev  = new double[d]; 
    ww_prev = new double[d];
    
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters_outer + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	// Initiate w and x
	for (j = 0; j < d; j++) {
		w_prev[j] = 0;
		w_tilda[j] = w_prev[j];
        x_prev[j]  = w_prev[j]; 
        y_prev[j]  = w_prev[j]; 
	}
      
	// Outer loop
	for (k = 0; k < iters_outer; k++) {
        
		// Evaluate function value if output requested
		if (evalf == true) {		
			hist[k] = compute_function_value(w_tilda, Xt, y, n, d, lambda);			
		}

		// Compute the full gradient, mu  
		compute_full_gradient(Xt, w_tilda, y, mu, n, d, lambda);
                //compute_zeroth_full_gradient(Xt, w_tilda, y, mu, n, d, lambda);
		// Add gradient of l2 reg and initiate ww_prev
		for (j = 0; j < d; j ++){
			mu[j] += lambda * w_tilda[j];
            ww_prev[j] = 0;
		}
        
        // Initiate ww
        ww = 0;
        
		// Inner loop
		for (i = 0; i < iters_inner; i++) {
            
			long idx = dis(gen);
            
            // Update w_prev
            for (j = 0; j < d; j ++){
                w_prev[j] = x_prev[j]*tau + w_tilda[j]*0.5 + y_prev[j]*(0.5 - tau);
            }
            
			// Compute gradient of last inner iter
			compute_partial_gradient(Xt, w_prev, y, g_prev, n, d, lambda, idx);
			
			// Compute the gradient of last outer iter
			compute_partial_gradient(Xt, w_tilda, y, g_tilda, n, d, lambda, idx);

			// Add gradient of l2 reg
			for (j = 0; j < d; j ++){
				g_prev[j] += lambda * w_prev[j];
				g_tilda[j] += lambda * w_tilda[j]; 
                x_prev1[j] = x_prev[j];
			}
                              
			// Update x_prev and y_prev           
            for (j = 0; j < d; j ++){
                x_prev[j] = x_prev[j] - alpha * (g_prev[j] - g_tilda[j] + mu[j]);
                y_prev[j] = w_prev[j] + tau * (x_prev[j] - x_prev1[j]);
                ww_prev[j] += y_prev[j]*pow((double)tau1, (double)i);
            }       
            ww += pow((double)tau1,(double)i);
		}
        
		for (j = 0; j < d; j++) {
			w_tilda[j] = ww_prev[j]/ww;
		}
    }
   
    hist[(int)(iters_outer)] = compute_function_value(w_tilda, Xt, y, n, d, lambda); 

    
	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] mu;
	delete[] g_prev;
	delete[] g_tilda;
	delete[] w_prev;
	delete[] w_tilda;
    delete[] x_prev;
    delete[] y_prev;
    delete[] ww_prev;
    
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
		plhs[0] = Katyusha(nlhs, prhs);
	}
	else {
		plhs[0] = Katyusha(nlhs, prhs);
	}
}
