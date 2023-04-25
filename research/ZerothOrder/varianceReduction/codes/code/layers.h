
#ifndef LAYERS_H
#define LAYERS_H

#include<cstdio>
#include<cmath>

#include<armadillo>
using namespace arma;


double least_forward(const mat& a, const mat& y){
	int dim1 = a.n_rows;
	double loss = 0;
	mat tmp(a);
	
	tmp = pow(a-y,2);
	loss = accu(tmp) / dim1 * 0.5;

	return loss;
}

void least_backward(const mat& a, const mat&  y, mat &grad){
	int dim1 = a.n_rows;
	grad = (a - y) / dim1;
}

double logistic_forward(const mat& a, const mat& y0){
	mat tmp = -a % y0;
	double loss;
	tmp = log(1+exp(tmp));
	loss = accu(tmp) / a.n_rows;
	return loss;
}


void logistic_backward(const mat& a, const mat&  y, mat &grad){
	int dim1 = a.n_rows;
	mat tmp = -a % y;
	grad = -pow(1+exp(tmp),-1) % y / dim1;
}

// softmax loss
double softmax_forward(const mat&  a, const mat&  y, mat &probs){
	int dim1 = a.n_rows;
	int dim2 = a.n_cols;
	double loss = 0;
	
	probs = exp(a);
	mat sums = repmat(sum(probs, 1), 1, dim2);
	probs = probs / sums;

     
	//compute loss
	int label = 0;
	for(int i=0; i<dim1; i++){
		label = (int)y(i,0);	
		loss += -log(probs(i,label))/dim1;
	}

	return loss;
}

void softmax_backward(const mat& a, const mat&  y, const mat&  probs, mat &grad){
	int dim1 = a.n_rows;
	int label = 0;
	grad = probs;

	for(int i=0; i<dim1; i++){
		label = y(i,0);
		grad(i,label) = grad(i,label)-1;
	}
	grad = grad / dim1;
}


void sigmoid_forward(const mat&  o, mat  &a){

	a = pow(1+ exp(-o),-1);
}


void sigmoid_backward(const mat& o, const mat& dl, mat &grad){

	grad =   dl % o % (1-o);
}

void relu_forward(const mat& o, mat &a){
	a = o;
	a.for_each([](mat::elem_type& val){ val=(val>0?val:0);});
}

void relu_backward(const mat& o, const mat& dl, mat &grad){
	grad = o;
	grad.for_each([](mat::elem_type& val){ val=(val>0?1:0);});
	grad = dl % grad;
}

void affine_forward(const mat& x, const mat& w, mat &o){
	o = x*w;
}


void affine_backward(const mat& x, const mat& w, const mat& dl,\
			   	mat &dx, mat &grad){

	grad = x.t()*dl;
	dx = dl * w.t();
}

#endif
