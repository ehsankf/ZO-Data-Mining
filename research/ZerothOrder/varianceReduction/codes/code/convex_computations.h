
#ifndef CONVEX_COMTATIONS_H
#define CONVEX_COMTATIONS_H

#include<cstdlib>
#include<cstdio>
#include<armadillo>

#include "layers.h"

using namespace arma;

// for different layers nn, we have to edit this file.
mat vec_2_mat(const std::vector<double>& w, int begin, int row, int col){
	mat out(row,col);
	int index = 0;
	for(int i=0; i<row; i++)
		for(int j=0; j<col; j++){ 
			index = i*col + j; 
			out(i,j) = w[begin+index];
	   	}
	return out;
}


std::vector<double> mat_2_vec(const mat& Matrix){
	int row = Matrix.n_rows;
	int col = Matrix.n_cols;
	std::vector<double> out(row*col, 0);
	for(int i=0; i<row; i++)
		for(int j=0; j<col; j++)
		out[i*col+j] = Matrix(i,j); 

	return out;
}


// compute regularization
double com_reg(const mat& w, double C){
	double reg = 0;
	reg = 0.5 * C * accu(pow(w,2));
	return reg;
}

void com_grad(const mat& x, const mat& y, const std::vector<double>& w, std::vector<double> &grad,\
			   	const std::vector<int>& dims, double C, int left, int right){
	
	// convert vector w to matrix. 3 layers in this case. 2 weight parameters.  
	//int num_layers = dims.size();
	mat w_1(dims[0], dims[1]);
	mat grad_1(dims[0], dims[1]);
	int begin=0;
	// w_1 conversion
	w_1 = vec_2_mat(w, begin, dims[0], dims[1]);

	int size = right - left;
	mat xb(size, dims[0]);
	mat yb(size, dims[1]);
	mat dx(size, dims[0]);
	mat o_1(size, dims[1]);
	mat dldo_1(size, dims[1]);

	xb = x.cols(left, right-1).t();
	yb = y.rows(left, right-1);

	// forward
	affine_forward(xb, w_1, o_1);
	least_forward(o_1, yb);

	// backward
	least_backward(o_1, yb, dldo_1);
	affine_backward(xb, w_1, dldo_1, dx, grad_1);

	// convert mat 2 vec
	grad = mat_2_vec(grad_1);

	// add regularization. 
	for(std::size_t i=0; i<w.size(); i++)
		grad[i] += C*w[i];
}

// compute objective value
double com_obj(const mat& x, const mat& y, const std::vector<double>& w, const std::vector<int>& dims, double C, int start, int end, double & acc){
	double obj = 0; 
	mat w_1(dims[0], dims[1]);
	// w_1 conversion
	int begin=0;
	w_1 = vec_2_mat(w, begin, dims[0], dims[1]);

	int size = end - start;
	mat xb(size, dims[0]);
	mat yb(size, dims[1]);
	xb = x.cols(start, end-1).t();
	yb = y.rows(start, end-1);

	mat o_1(size, dims[1]);
	mat a_1(size, dims[1]);
	
	// forward
	affine_forward(xb, w_1, o_1);
	obj = least_forward(o_1, yb);

	acc = std::pow(obj * 2, 0.5);

	obj += com_reg(w_1, C);

	return obj;
}
	
void com_numerical_grad(const mat& x, const mat& y, std::vector<double> w, std::vector<double> &grad,\
			   	const std::vector<int>& dims, double C, int left, int right, int coord_left, int coord_right, double epsilon){
	double acc;
	for(int i = coord_left; i < coord_right; i++) {
		std::vector<double> w_1(w);
		std::vector<double> w_2(w);
		w_1[i] -= epsilon;
		w_2[i] += epsilon;

		double obj2 = com_obj(x, y, w_2, dims, C, left, right, acc);
		double obj1 = com_obj(x, y, w_1, dims, C, left, right, acc);
		grad[i] = (obj2 - obj1) / (coord_right - coord_left) * w.size() / 2 / epsilon;
	}
}

#endif
