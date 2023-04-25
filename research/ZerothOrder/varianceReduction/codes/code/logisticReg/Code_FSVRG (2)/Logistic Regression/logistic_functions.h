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

/// Compute the function value of average regularized logistic loss
double compute_function_value(double* w, double *Xt, double *y,
	long n, long d, double lambda)
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
		value += (lambda / 2) * w[j] * w[j];
	}
	return value;
}
