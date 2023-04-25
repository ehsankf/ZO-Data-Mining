#include <iostream>
#include <algorithm>
#include <vector>
#include "mex.h"

using namespace std;


double vector_dot_product(double *x, double *y, long d) {
	long i;
	double ret = 0;

	for (i = 0; i < d; i++) {
		ret += x[i] * y[i];
	}

	return ret;
}

double* compute_full_gradient(double *w, double *Xt, double *y, long n, long d) {
	long i, j;
	double tmp;
	double *g = new double[d];

	for (i = 0; i < d; i++) {
		g[i] = 0;
	}

	for (i = 0; i < n; i++) {
		tmp = vector_dot_product(Xt + i*d, w, d);
		if (tmp * y[i] < 1) {
			for (j = 0; j < d; j++) {
				g[j] -= y[i] * Xt[i*d + j]/n;
			}
		}
	}

	return g;
}

double* compute_subgradient(double *w, double *X, double y, long d) {
	long i;
	double *g = new double[d];

	for (i = 0; i < d; i++) {
		g[i] = 0;
	}

	if (y * vector_dot_product(X, w, d) < 1) {
		for (i = 0; i < d; i++) {
			g[i] = -y * X[i];
		}
	}

	return g;
}

double* compute_batch_gradient(double *w, double *Xt, double *y, long mb, long d) {
	long i, j, counter;
	double tmp;
	double *g = new double[d];

	counter = 0;

	for (i = 0; i < d; i++) {
		g[i] = 0;
	}

	for (i = 0; i < mb; i++){ 
		tmp = vector_dot_product(Xt + i*d, w, d);
		if (tmp*y[i] < 1) {counter++;}
	}

	if (counter > 0) {
		for (j = 0; j < d; j++) {
				g[j] -= y[i] * Xt[i*d + j] / mb;
		}
	}

	return g;
}

void update_point(double *w, double eta, double gamma, double *g, long d) {
	long i;
	for (i = 0; i < d; i++) {
		w[i] = w[i]*(1 - eta*gamma) - g[i]*eta;
	}
}