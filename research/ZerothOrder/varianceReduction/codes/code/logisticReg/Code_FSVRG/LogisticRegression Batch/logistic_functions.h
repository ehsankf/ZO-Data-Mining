// Without otherwise specify:

/// *Xt - data matrix
/// *y  - set of labels
/// n   - number of sample
/// d   - dimension of the data
/// lambda - regularization parameter
/// *w - test point


/// compute_sigmoid computes the derivative of logistic loss,
/// i.e. ~ 1 / (1 + exp(x))
/// *x - pointer to the first element of the training example
///		 e.g. Xt + d*i for i-th example
#include <math.h>


double comp_l1_norm(double* vec, long d) {
    double res = 0.0;
    for(size_t i = 0; i < d; i ++){
        res += abs(vec[i]);
    }
    return res;
}

double compute_sigmoid(double *x, double *w, double y, long d)
{
	double tmp = 0;
	// Inner product
	for (long j = 0; j < d; j++) {
		tmp += w[j] * x[j];
	}
	tmp = exp(-y * tmp);
	tmp = 1 / (1 + tmp);
	return tmp;
}


/// Compute the function value of average regularized logistic loss

double compute_function_value(double* w, double *Xt, double *y,

	long n, long d, double lambda, double lambda1)

{

	double value = 0;

	double tmp;

	// Compute losses of individual functions and average them

	for (long i = 0; i < n; i++) {

		tmp = 0;

		for (long j = 0; j < d; j++) {
			tmp += Xt[i*d + j] * w[j];
		}
		value += log(1 + exp(-y[i] * tmp));
	}
	value = value / n;

	// Add regularization term
	for (long j = 0; j < d; j++) {
		value += (lambda / 2) * w[j] * w[j] + lambda1 * abs(w[j]);
	}
	return value;
}

double compute_zeroth_function_value(double* w, double *Xt, double *y,

	long n, long d, double lambda, double lambda1)

{

	double value = 0;

	double tmp;

	// Compute losses of individual functions and average them

	for (long i = 0; i < n; i++) {

		tmp = 0;

		for (long j = 0; j < d; j++) {
			tmp += Xt[i*d + j] * w[j];
		}
		value += log(1 + exp(-y[i] * tmp));
	}
	value = value / n;

	// Add regularization term
	/*for (long j = 0; j < d; j++) {
		value += (lambda / 2) * w[j] * w[j] + lambda1 * abs(w[j]);
	}*/
	return value;
}

/// Compute the function value of logistic loss on batch
double compute_partial_function_value(double* w, double *Xt, double *y,
	long n, long d, double lambda, long left, long right)
{
	double value = 0;
	
	// Compute losses of individual functions 
      for(int i = left; i < right; i++) {
        double tmp = 0;
        
	for (long j = 0; j < d; j++) {
		tmp += Xt[i*d + j] * w[j];
        }
	value += log(1 + exp(-y[i] * tmp));
      }	
	return value;
}
/// compute_full_gradient computes the gradient 
/// of the entire function. Gradient is changed in place in g
/// *g - gradient; updated in place; input value irrelevant
void compute_full_gradient(double *Xt, double *w, double *y, double *g,
	long n, long d, double lambda)
{
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
	}

	// Sum the gradients of individual functions
	double sigmoid;
	for (long i = 0; i < n; i++) {
		sigmoid = compute_sigmoid(Xt + d*i, w, y[i], d);
		for (long j = 0; j < d; j++) {
			g[j] += (sigmoid - 1) * y[i] * Xt[d*i + j];
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
	}
}


void compute_zeroth_full_gradient(double *Xt, double *w, double *y, double *g,
	long n, long d, double lambda, double lambda1, long k)
{
        
        double *w1, *w2;  
        double epsilon =  (float) 1e-5/k;
        //printf("Epsilon %f %f", k, epsilon);
        w1  = new double[d];
        w2  = new double[d];
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
                w1[i] = w[i];
                w2[i] = w[i];
	}
        for(int i = 0; i < d; i++) {
		w1[i] -= epsilon;
		w2[i] += epsilon;

		double obj2 = compute_zeroth_function_value(w2, Xt, y, n, d, lambda, lambda1);
		double obj1 = compute_zeroth_function_value(w1, Xt, y, n, d, lambda, lambda1);
		g[i] = (obj2 - obj1) / (2 * epsilon);
                w1[i] += epsilon;
		w2[i] -= epsilon;
                if (w1[i] != w2[i])
                    printf("i=%d w1[i] = %d, w2[i] = %dWrong\n", i, w1[i], w2[i]);
               
	}
	// Sum the gradients of individual functions
	/*double sigmoid;
	for (long i = 0; i < n; i++) {
		sigmoid = compute_sigmoid(Xt + d*i, w, y[i], d);
		for (long j = 0; j < d; j++) {
			g[j] += (sigmoid - 1) * y[i] * Xt[d*i + j];
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
	}
        for(int i = coord_left; i < coord_right; i++) {
		std::vector<double> w_1(w);
		std::vector<double> w_2(w);
		w_1[i] -= epsilon;
		w_2[i] += epsilon;

		double obj2 = com_obj(x, y, w_2, dims, C, left, right, acc);
		double obj1 = com_obj(x, y, w_1, dims, C, left, right, acc);
		grad[i] = (obj2 - obj1) / (coord_right - coord_left) * w.size() / 2 / epsilon;
	}*/
        //compute_function_value(w, Xt, y, n, d, lambda);
        delete[] w1;
        delete[] w2;
}

/// compute_partial_gradient computes the gradient
/// of one sample. Gradient is changed in place in g
/// *g - gradient; updated in place; input value irrelevant
/// i - selected column
void compute_partial_gradient(double *Xt, double *w, double *y, double *g,
	long n, long d, double lambda, long i)
{
	// Initialize the gradient
	for (long j = 0; j < d; j++) {
		g[j] = 0;
	}

	// Sum the gradients of individual functions
	double sigmoid = compute_sigmoid(Xt + d*i, w, y[i], d);
	for (long j = 0; j < d; j++) {
		g[j] += (sigmoid - 1) * y[i] * Xt[d*i + j];
	}
}

void compute_zeroth_partial_gradient(double *Xt, double *w, double *y, double *g,

	long n, long d, double lambda, double lambda1, long left, long right, long k)

{

        double *w1, *w2;  
        double epsilon =  (float) 1e-9/k;
        long size = right - left;
        //printf("Epsilon %f %f", k, epsilon);
        w1  = new double[d];
        w2  = new double[d];
	// Initialize the gradient
	for (long j = 0; j < d; j++) {
		g[j] = 0;
                w1[j] = w[j];
                w2[j] = w[j];
	}
        
        for(int j = 0; j < d; j++) {
		w1[j] = w[j] - epsilon;
		w2[j] = w[j] + epsilon;

                if (w1[j] - w2[j] > 2 * epsilon || w1[j] - w2[j] == 0)
                    printf("partial j=%d w1[j] = %d, w2[j] = %d Not\n", j, w1[j], w2[j]);

		double obj2 = compute_partial_function_value(w2, Xt, y, n, d, lambda, left, right);
		double obj1 = compute_partial_function_value(w1, Xt, y, n, d, lambda, left, right);
		g[j] += (obj2 - obj1) / (2 * epsilon);
                w1[j] = w[j];
		w2[j] = w[j];
                if (w1[j] != w2[j]){
                    printf("partial j=%d w1[j] = %d, w2[j] = %d, w[j] = %d Wrong\n", j, w1[j], w2[j], w[j]);
                    printf("DDDDDDDDDDDDD");
                }
       
          
	}
        
        /*printf("Size %d \n", size);
        for (long j = 0; j < d; j++) {
		g[j] = g[j] / size;
	}*/

	// Initialize the gradient

	/*for (long j = 0; j < d; j++) {

		g[j] = 0;

	}



	// Sum the gradients of individual functions

	double sigmoid = compute_sigmoid(Xt + d*i, w, y[i], d);

	for (long j = 0; j < d; j++) {

		g[j] += (sigmoid - 1) * y[i] * Xt[d*i + j];

	}*/
    delete[] w1;
    delete[] w2;

}



