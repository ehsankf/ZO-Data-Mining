/*Description
 * This is for parallel SVRG on multicore machine. 
 *  * Author: Zhouyuan Huo 
 *  * Time: 2016-05-06
 * neural network*/

// need to use: export OPENBLAS_NUM_THREADS=1 before we run!

#include<cstdlib>
#include<cstdio>
#include<omp.h>
#include<cmath>
#include<algorithm> 
#include<ctime> 
#include<vector> 
#include<string>
#include "layers.h"
#include "data.h"
#include "convex_computations.h"
#include "defines.h"

#include<armadillo>

using namespace arma;

int main(int argc, char* argv[]){
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	const int Num_th = FLAGS_n_threads;
	const int N  = FLAGS_num_train;
	const int N_test = FLAGS_num_test;
	const int d1 = FLAGS_dim_1; 
	const int d2 = FLAGS_dim_2;
	const int batch_size = FLAGS_mini_batch;
	const int batch_coord = FLAGS_coord_block;
	const int  epochs = FLAGS_max_epoch;
	const double eta = FLAGS_learning_rate_dec;  // step = lr / (1+epoch)^eta
	const int In_iters = FLAGS_inner_iter; // inner loop epoch
	const double lr = FLAGS_learning_rate;  //learning rate
	const double C = FLAGS_weight_decay;  // reg 
	const double epsilon = FLAGS_epsilon;

	mat x(N, d1);
	mat y(N, d2);
	mat x_test(N_test, d1);
	mat y_test(N_test, d2);
	mat w_1(d1, d2);
	std::vector<int> dims;	
	dims.push_back(d1);
	dims.push_back(d2);

		std::vector<double> timearray(epochs+1,0);

		int  para_size = d1*d2;
		std::vector<double> w;
		std::vector<double> u(para_size, 0);
		std::vector<double> full_grad(para_size, 0);

		double step, timer, L2norm;
		int  epoch;	
		double start;


		//initialization dataset.
		//initialize x,y
		//init_convex_data(x, y);
		//load_data(x, y);
		
		x.load("x.bin", arma::raw_binary);
		y.load("r.bin", arma::raw_binary);	

		x.reshape(x.n_elem/d1, d1);
		x_test = x.rows(N, x.n_rows-1);
		y_test = y.rows(N, y.n_rows-1);
		x_test = x_test.t();

		x = x.rows(0, N-1);
		y = y.rows(0, N-1);
		x = x.t();


		// store w_1 to w.
	    w = mat_2_vec(w_1);

	std::printf("epoch  train_loss  train_acc trian_rmse  test_loss  test_acc  test_rmse  train_L2norm  Time  \n");

	int num_batches_train = N / batch_size;
	int num_batches_coord = para_size / batch_coord;
	int train_per_thread = N / Num_th;

	omp_set_num_threads(Num_th);

	#pragma omp parallel
	{
		pin_to_core(omp_get_thread_num()+FLAGS_base_thread);
	}

		// update
		for(epoch=0; epoch<epochs; epoch++){

			srand(epoch);
			u = w;

			// update step size
			step = 1.0 * lr / pow(1 + epoch, eta);
			if (epoch==0) step = FLAGS_first_epoch_learning_rate;

			// evaluate stage 
			// evaluate training objective value
			double train_obj=0, train_acc=0, test_obj=0, test_acc=0;

			#pragma omp parallel for  reduction(+:train_acc, train_obj) 
			for(int j=0; j<Num_th; j++){
				int left = j * train_per_thread;
				int right = (j+1) * train_per_thread;
				if ( j == Num_th - 1) right = N;
				double tmp_acc=0, tmp_obj;
				tmp_obj = com_obj(x, y, w, dims, C, left, right, tmp_acc);
				train_acc += tmp_acc * (right - left) / N; 
				train_obj += tmp_obj * (right - left) / N; 
			}


			// evaluate L2norm
			L2norm = 0;
			std::vector<double> sum_grad(para_size,0);
			#pragma omp parallel for
			for(int j=0; j<Num_th; j++){
				std::vector<double> grad(para_size,0);
				//std::vector<double> grad_1(para_size,0);
				int left = j * train_per_thread;
				int right = (j+1) * train_per_thread;
				if ( j == Num_th - 1) right = N;
				com_grad(x, y, w, grad, dims, C, left, right);

				#pragma omp critical
				for(int i=0; i<para_size; i++) {
					sum_grad[i] += grad[i] * (right - left) / N;
				}
			}

			#pragma omp parallel for reduction(+:L2norm)
			for(int i=0; i<para_size; i++) {
				L2norm += sum_grad[i]*sum_grad[i];
			}
			

			// compute train rmse
			double train_rmse = 0, test_rmse = 0;
			mat w1(w);
			w1.resize(d1, d2);
			train_rmse = sqrt(mean(square(y - x.t() * w1)))[0];
			test_rmse = sqrt(mean(square(y_test - x_test.t() * w1)))[0];

			std::printf("%d  %.15f  %.15f  %.15f  %.15f   %.15f   %.15f  %.15f  %.2f\n ", epoch, train_obj, train_acc, train_rmse, test_obj, test_acc, test_rmse, L2norm, timearray[epoch]);

			start = omp_get_wtime();

			// update stage
			if(epoch != 0) {
			std::fill(full_grad.begin(), full_grad.end(), 0);
			#pragma omp parallel for 
			for(int j=0; j<Num_th; j++){
				std::vector<double> grad(para_size,0);
				int left = j * train_per_thread;
				int right = (j+1) * train_per_thread;
				if ( j == Num_th - 1) right = N;
				com_numerical_grad(x, y, w, grad, dims, C, left, right, 0, para_size, epsilon);
				//com_grad(x, y, w, grad, dims, C, left, right);

				#pragma omp critical
				for(int i=0; i<para_size; i++)
					full_grad[i] += grad[i] * (right - left) / N;
			}
			}

			#pragma omp parallel for 
			for(int j=0; j<In_iters; j++){
				std::vector<double> grad(para_size, 0);
				std::vector<double> grad_u(para_size, 0);

				//compute gradient.
				int train_index = rand() % num_batches_train;
				int left = train_index * batch_size;
				int right = (train_index + 1) * batch_size;
				if (train_index == num_batches_train-1) 
					right = N;

				int coord_index = rand() % num_batches_coord;
				int coord_left = batch_coord * coord_index;
				int coord_right = batch_coord * (1 + coord_index);
				if (coord_index == num_batches_coord - 1) 
					coord_right = para_size;

				com_numerical_grad(x, y, w, grad, dims, C, left, right, coord_left, coord_right, epsilon);
				if (epoch != 0) com_numerical_grad(x, y, u, grad_u, dims, C, left, right, coord_left, coord_right, epsilon);

				//update w.
				for(int i=coord_left; i<coord_right; i++){
					w[i] -= step * (grad[i] - grad_u[i] + full_grad[i]/(coord_right-coord_left)*d1); 
				}
			}

			timer = omp_get_wtime() - start;
			timearray[epoch+1] = timer + timearray[epoch];

		}

	return 0;
}


