
void compute_full_gradient(double *Xt,
                           double *w,
                           double *y, 
                           double *g,
                           long n, 
                           long d, 
                           double lambda)
{
    // Init the gradient
    for (long i = 0; i < d; i++) {
        g[i] = 0;
    }

    for (long i = 0; i < n; i++) {
        // Xt(i) dot w;
        double x_i_w = 0;
        for (long j = 0; j < d; j ++){
            x_i_w += Xt[i*d + j] * w[j];
        }
        // Xt(i) dot w - y
        double x_i_w_minus_y = (x_i_w - y[i]);

        for (long j = 0; j < d; j ++){
            g[j] += x_i_w_minus_y * Xt[i*d + j];
        }
    }

    // Add gradient of regularizer
    for (long i = 0; i < d; i++) {
        g[i] = g[i] / n;
    }
}

void compute_partial_gradient(double *Xt,
                              double *w, 
                              double *y, 
                              double *g,
                              long n,
                              long d, 
                              double lambda, 
                              long i)
{
    // Init the gradient
    for (long k = 0; k < d; k++) {
        g[k] = 0;
    }

    // Xt(i) dot w;
    double x_i_w = 0;
    for (long j = 0; j < d; j ++){
        x_i_w += Xt[i*d + j] * w[j];
    }
    // Xt(i) dot w - y
    double x_i_w_minus_y = (x_i_w - y[i]);

    for (long j = 0; j < d; j ++){
        g[j] += x_i_w_minus_y * Xt[i*d + j];
    }
}


double compute_function_value(double* w, 
                              double *Xt, 
                              double *y,
                              long n, 
                              long d, 
                              double lambda)
{
    double value = 0;

    for (long i = 0; i < n; i++) {
        // Xt(i) dot w;
        double x_i_w = 0;
        for (long j = 0; j < d; j ++){
            x_i_w += Xt[i*d + j] * w[j];
        }
        // Xt(i) dot w - y
        double x_i_w_minus_y = x_i_w - y[i];

        // (Xt(i) dot w - y) ^ 2
        value += x_i_w_minus_y * x_i_w_minus_y;
    }
    
    // new
	value = value/(2*n);  

	for (long j = 0; j < d; j++) {
		value += (lambda / 2) * w[j] * w[j];
	}
    
    return value;
}


double compute_saga_gradient(double *Xt,
                              double *w, 
                              double *y, 
                              long n,
                              long d, 
                              long i)
{
    double g1_value = - y[i];
    for(long j = 0; j < d; j ++) {
        g1_value +=  w[j] * Xt[i*d + j];
    }
    return g1_value;
}
