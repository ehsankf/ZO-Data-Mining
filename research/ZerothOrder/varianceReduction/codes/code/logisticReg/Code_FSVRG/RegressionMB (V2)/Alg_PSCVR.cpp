#include <math.h>
#include "mex.h"
#include <string.h>
#include "logistic_functions.h"
#include "utilities.h"
#include <random>
#include <iostream>
#include <ctime>

using namespace std;
unsigned long long int query_count = 0;
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
	long n, d, minibatch_size, Batch_size; // Dimensions of problem
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
        double *funcvalue, *quercount, *timecount;
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
        Batch_size = mxGetScalar(prhs[8]);    
        clock_t c_start, c_end;
 
	if (nlhs == 1) {
		evalf = true;
	}
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
        long num_batches_train = (int) n / minibatch_size;
        int num_Batches = (int) n / Batch_size;
        printf("num_batches_train %d", num_batches_train);
	iters_inner = 2 * n; // Number of inner iterations
	
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_batches_train-1);
    //std::uniform_int_distribution<> disBatch(0, num_Batches-1);

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	mu = new double[d];
        mu1 = new double[d];
	g_prev  = new double[d];
	g_tilda = new double[d];
	w_prev  = new double[d];
        funcvalue = new double[(int)(iters_outer+1)];
        quercount = new double[(int)(iters_outer+1)];
        timecount = new double[(int)(iters_outer+1)];
        //ww_prev = new double[d];
	w_tilda = new double[d];
        Matrix = new double*[num_batches_train];
        for (i = 0; i < num_batches_train; i++)
           Matrix[i] = new double[d];
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(3 * iters_outer + 3, 1, mxREAL);
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

        for (j = 0; j <= iters_outer; j++) {
		funcvalue[j] = 0;
		quercount[j] = 0;
                timecount[j] = 0;
	}

        //compute_zeroth_full_gradient(Xt, w_tilda, y, mu, n, d, lambda);
        printf("Hii");
        
        int Batch_index = rand() % num_Batches;
	int left_Batch = Batch_index * Batch_size;
	int right_Batch = (Batch_index + 1) * Batch_size;
        if (Batch_index == num_Batches-1) 
	      right_Batch = n;
         
	// Outer loop
	for (k = 0; k < iters_outer; k++) {

                
        
		// Evaluate function value if output requested
		if (evalf == true) {		
                        funcvalue[k] = compute_function_value(w_tilda, Xt, y, n, d, lambda, lambda1);
                        quercount[k] = query_count;
			//hist[k] = compute_function_value(w_tilda, Xt, y, n, d, lambda, lambda1);
                        			
		}
                //srand(k);
                //if(k%5 == 0){
                   srand(time(NULL));
                   int xrand = rand();
                   xrand +=k;
                   int Batch_index = xrand % num_Batches; //dis(gen) % num_Batches;;//rand() % num_Batches;
		   int left_Batch = Batch_index * Batch_size;
		   int right_Batch = (Batch_index + 1) * Batch_size;
                   if (Batch_index == num_Batches-1) 
		       right_Batch = n; 
                   int size_Batch = right_Batch - left_Batch;
                 
                printf("Itr %d BatchIndex %d num_Batches %d minibatch_size %d query_count %llu \n", k, Batch_index, num_Batches, minibatch_size, query_count); //}

		// mu with the full gradient 
		//compute_full_gradient(Xt, w_tilda, y, mu1, n, d, lambda);
                compute_zeroth_Batch_full_gradient(Xt, w_tilda, y, mu, n, d, lambda, lambda1, left_Batch, right_Batch, k+1);
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
                c_start = std::clock();
		// Inner loop
		for (i = 0; i < iters_inner; i++) {

                        int num_batches_train = (int) size_Batch / minibatch_size;
                        int train_index = rand() % num_batches_train;
			int left = left_Batch + train_index * minibatch_size;
			int right = left_Batch + (train_index + 1) * minibatch_size;
                        if (train_index == num_batches_train-1) 
					right = right_Batch;

                        /*long train_index = dis(gen);//rand() % num_batches_train;
			long left = train_index * minibatch_size;
			long right = (train_index + 1) * minibatch_size;
                        
			if (train_index == num_batches_train-1) 
					right = n;*/
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
            c_end = std::clock();
		for (j = 0; j < d; j ++) {
            w_tilda[j] = w_prev[j];
            //w_tilda[j] = ww_prev[j]/iters_inner;
            //w_prev[j] = w_tilda[j];
		}
     if(k==0)
       timecount[k] = c_end - c_start;
     else 
       timecount[k] = c_end - c_start + timecount[k-1];
    }
    c_end = std::clock();
    timecount[(int)(iters_outer)] = c_end - c_start + timecount[(int)(iters_outer-1)];
    if (evalf == true) {
        funcvalue[(int)(iters_outer)] = compute_function_value(w_tilda, Xt, y, n, d, lambda, lambda1);
        printf("query_count %llu \n", query_count);
        quercount[(int)(iters_outer)] = query_count;
        //hist[(int)(iters_outer)] = compute_function_value(w_tilda, Xt, y, n, d, lambda, lambda1); //new
       
    }
    for (k = 0; k <= iters_outer; k++) 
          hist[k] = funcvalue[k];
    for (k = iters_outer+1; k <= 2 * iters_outer+1; k++) 
          hist[k] = quercount[(int)(k-iters_outer-1)];
    for (k = 2 * iters_outer+2; k <= 3 * iters_outer+2; k++) 
          hist[k] = timecount[(int)(k-2*iters_outer-2)];
    
    
	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] mu;
	delete[] g_prev;
	delete[] g_tilda;
	delete[] w_prev;
	delete[] w_tilda;
        delete[] Matrix;
        delete[] funcvalue;        
        delete[] quercount;
        delete[] timecount;
        query_count = 0;

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
