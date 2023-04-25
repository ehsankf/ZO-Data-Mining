#include <cmath>
#include <math.h>
#include "mex.h"
#include <string.h>
#include "svm_dense.h"
#include <random>
#include <ctime>
#include <iostream>
using namespace std;

/*
	USAGE:
	hist = Alg_FSVRG(w, Xt, y, gamma, max_it, mb);
	mb is optional
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; for output
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
*/

inline double round(double x) { return (floor(x + 0.5)); }

mxArray* FSVRG_dense(int nlhs, int nrhs, const mxArray *prhs[]) {

	double * Xt 	= mxGetPr(prhs[0]); // Sample matrix (transposed)
	double * y 		= mxGetPr(prhs[1]); // Labels
	double gamma 	= mxGetScalar(prhs[2]);
	long max_it 	= mxGetScalar(prhs[3]);
	long mb 		= 0;
	if (nrhs < 5) mb = 1; // This is an non mini batch version
	else mb 		= mxGetScalar(prhs[4]);
	double theta  	= 0.10; 

	// Xt will be transposed when passing in
	long d 			= mxGetM(prhs[0]); // Number of features, or dimension of problem
	long n 			= mxGetN(prhs[0]); // Number of samples, or data points
	long max1 		= round(max_it/(2*n)) + 1;
	long m 			= round(2*n/mb);
	double eta 		= 0.75; 
    
	srand ( unsigned ( std::time(0) ) );
	vector<long> rnd_pm;
	for (long i=0; i<n; ++i) rnd_pm.push_back(i);
	random_shuffle ( rnd_pm.begin(), rnd_pm.end() );


	double * x 		= new double[d];
	double * w 		= new double[d];
	double * xold 	= new double[d];
	mxArray * plhs 	= mxCreateDoubleMatrix(d, 1, mxREAL);
	double * ret 	= mxGetPr(plhs);

	for (long j = 0; j < d; j++) {
		xold[j] = 0;
	}
    
    double lambda = 1 - eta*gamma; 
    
	//////////////////////////////////////////////////////////////////
	///         FSVRG        /////////////////////////////////////////
	//////////////////////////////////////////////////////////////////
	// The outer loop
	for (long k = 0; k < max1; k++) {
        
		double * gold = compute_full_gradient(xold, Xt, y, n, d);
		double *ww = new double[d];
		for(long j = 0; j < d; j ++) {
            ww[j] = 0;
            w[j]  = xold[j];
        }

		// The inner loop
		for (long i = 0; i < m; i++) {
			double * gg;
			double * ggold;
			if (mb == 1){
				
                long idx = rnd_pm[rand()%n];
                
				gg = compute_subgradient(w, Xt + idx*d, y[idx], d);
				if (k != 0){
					ggold = compute_subgradient(xold, Xt + idx*d, y[idx], d);
					for (long j = 0; j < d; j++) {
						gg[j] += gold[j] - ggold[j];
					}
					delete [] ggold;			
				}				
			} else {
				mexPrintf("This is an non mini batch version");
				return 0;
			}

			for (long j = 0; j < d; j++) {
				x[j] = lambda * x[j] - eta * gg[j];
                w[j] = xold[j] + theta * (x[j] - xold[j]);
                ww[j] += w[j];
			}
			delete [] gg;
		}

		for (long j = 0; j < d; j++) {
            xold[j] = ww[j] / m;
			x[j] = xold[j];
		}
		delete [] gold;
	}

	for (long j = 0; j < d; j++) {
        ret[j] = x[j];
    }

	delete[] x;
	delete[] w;
	delete[] xold;

	return plhs;
}

/// nlhs - 	  number of output parameters
/// *plhs[] - array poiters to the outputs
/// nrhs - 	  number of input parameters
/// *prhs[] - array of pointers to inputs
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (mxIsSparse(prhs[0])) {
		cout << "FSVRG_Sparse is not ready" << endl;
		// plhs[0] = FSVRG_sparse(nlhs, prhs);
	}
	else {
		plhs[0] = FSVRG_dense(nlhs, nrhs, prhs);
	}
}
