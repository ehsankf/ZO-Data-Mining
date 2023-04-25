/*
 * Initialize data
 */

#ifndef DATA_H
#define DATA_H

#include<string>
#include<cstdlib> 
#include<cstdio>
#include<time.h>
#include<fstream>
#include<algorithm>
#include<cassert>
#include<string.h>
#include<random>

#include<armadillo>

using namespace arma;

void init_convex_data(mat &x, mat &y){
	int num = x.n_rows; 
	int dim1 = x.n_cols;	
	int dim2 = y.n_cols;	

	mat w(dim1,dim2);

	srand(0);
	/*
	for(int i=0; i<dim1; i++)
	for(int j=0; j<dim2; j++) {
		w(i, j) = (1.0*rand()/RAND_MAX - 0.5);
	}
	*/

	for(int i=0; i<num; i++) {
	for(int j=0; j<dim1; j++) {
		x(i,j) = rand() % 10;
	}
	}

	for(int i=0; i<num; i++) {
	for(int j=0; j<dim2; j++) {
		y(i,j) = rand() % 10;
	}
	}
	x =  x / 10;

}


void load_data(mat &x, mat &y){
	int num = x.n_rows; int dim1 = x.n_cols;	int dim2 = y.n_cols;	
	int i,j;
	double randx = 0;

	srand(0);
	for(i=0; i<num; i++){
		for(j=0; j<dim1; j++){
			randx = 1.0 * rand() / RAND_MAX;
			x(i,j) = randx;
		}

		for(j=0; j<dim2; j++){
			randx = 1.0 * rand() / RAND_MAX ;
			y(i,j) = randx;
		}
	}

}

// if seed !=0 initialize with seed random value. if flag == 0 initialize with  0.
void load_weight(mat &w, int seed, std::string full_path=""){
	int dim1 = w.n_rows;
	int dim2 = w.n_cols;
	int i,j;

	if (full_path.empty()){
		if(seed!=0){
			srand(seed);
			for(i=0; i<dim1; i++)
				for(j=0; j<dim2; j++){
					w(i, j) = (1.0*rand()/RAND_MAX - 0.5) * 2 / dim1;
					//w(i, j) = (1.0*rand()/RAND_MAX - 0.5) * 2;
				}
		}	
		else{
			for(i=0; i<dim1; i++)
				for(j=0; j<dim2; j++)
					w(i, j) = 0;
		}
	}
	else{
		std::ifstream file(full_path, std::ios::binary);
		if(file.is_open()){
			for(i=0; i<dim2; i++)
				for(j=0; j<dim1; j++){
					double temp = 0;
					file.read((char*)&temp, sizeof(double));
					w(j, i) = (double)temp;
				}	
		}	
		else
			printf("Can not open file!\n");
		file.close();
	}
}


// if seed !=0 initialize with seed random value. if flag == 0 initialize with  0.
void load_weight(std::vector<double> &w, int seed, std::string full_path=""){
	int dim = w.size();
	int i;

	if (full_path.empty()){
		srand(seed);
		for(i=0; i<dim; i++)
			w[i] = (1.0*rand()/RAND_MAX - 0.5) * 2 ;
	}
	else{
		std::ifstream file(full_path, std::ios::binary);
		if(file.is_open()){
			for(i=0; i<dim; i++){
				double temp = 0;
				file.read((char*)&temp, sizeof(double));
				w[i] = (double)temp;
			}
		}	
		else
			printf("Can not open file!\n");
		file.close();
	}
}


void save_weight(std::string full_path, const mat & w){
	std::ofstream file(full_path, std::ios::binary);
	
	if(file.is_open()){
		file.write((char*)&w(0), sizeof(double)*(int)w.n_elem);
	}
	else
		printf("Can not open file!\n");
	file.close();
}

void save_weight(std::string full_path, const std::vector<double> & w){
	std::ofstream file(full_path, std::ios::binary);
	if(file.is_open()){	
		file.write((char*)&w[0], sizeof(double)*(int)w.size());
	}
	else
		printf("Can not open file!\n");
	file.close();
}

		

#endif 
