#include <cmath>
#include <vector>
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
	hist = Alg_SGD(w, Xt, y, gamma, max_it, d_bound);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; for output
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
*/

mxArray* SGD_dense(int nlhs, int nrhs, const mxArray *prhs[]) {

	double * Xt    = mxGetPr(prhs[0]); // Sample matrix (transposed)
	double * y 	   = mxGetPr(prhs[1]); // Labels
	double gamma   = mxGetScalar(prhs[2]);
	long max_it    = mxGetScalar(prhs[3]);
	double d_bound = mxGetScalar(prhs[4]);

	// Xt will be transposed when passing in
	long d 		   = mxGetM(prhs[0]); // Number of features, or dimension of problem
	long n 		   = mxGetN(prhs[0]); // Number of samples, or data points

	srand ( unsigned ( std::time(0) ) );
	vector<int> rnd_pm;
	for (int i=0; i<n; ++i) rnd_pm.push_back(i);
	random_shuffle ( rnd_pm.begin(), rnd_pm.end() );

	double * x 	   = new double[d];
	double * x_avg = new double[d];
	mxArray * plhs = mxCreateDoubleMatrix(d, 1, mxREAL);
	double * ret   = mxGetPr(plhs);
	for (long j = 0; j < d; j++) { 
        x[j] = 0; 
        x_avg[j] = 0; 
    }

	for (long k = 1; k < max_it+1; k++) {
        
		int idx = rnd_pm[k%n];
        
		double eta = d_bound / sqrt((double)k);
		double * gg = compute_subgradient(x, Xt + idx*d, y[idx], d);
        
		for (long i = 0; i < d; i ++) {
			x[i] = (1-eta*gamma)*x[i] - eta*gg[i];
        }
		for (long i = 0; i < d; i ++) {
			x_avg[i] *= (k - 1);
			x_avg[i] += x[i];
			x_avg[i] /= k;			
		}
		delete[] gg;
	}
	
	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] x;
	delete[] x_avg;

	for (long j = 0; j < d; j++) {
        ret[j] = x_avg[j];
    }
	return plhs;
}

/// nlhs - 		number of output parameters
/// *plhs[] - 	array poiters to the outputs
/// nrhs - 		number of input parameters
/// *prhs[] - 	array of pointers to inputs

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (mxIsSparse(prhs[0])) {
		cout << "SGD sparse is not ready" << endl;
	}
	else {
		plhs[0] = SGD_dense(nlhs, nrhs, prhs);
	}
}
