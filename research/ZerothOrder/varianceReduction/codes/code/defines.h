
#ifndef _DEFINES_
#define _DEFINES_

#include <thread>
#include <gflags/gflags.h>

void pin_to_core(size_t core) {
#ifdef _GNU_SOURCE
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset); // clear set, so that it contains no cpu.
    CPU_SET(core, &cpuset); // add cpu  to cpuset
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

DEFINE_int32(base_thread, 0, "First threads to start.");
DEFINE_int32(n_threads, 1, "Number of threads.");
DEFINE_int32(num_train, 100, "Number of training datasets.");
DEFINE_int32(num_test, 1, "Number of testing datasets.");
DEFINE_int32(dim_1, 300, "dimmension 1 number.");
DEFINE_int32(dim_2, 1, "dimmension 2 number.");
DEFINE_int32(dim_3, 1, "dimmension 3 number.");
DEFINE_int32(dim_4, 1, "dimmension 4 number.");
DEFINE_int32(mini_batch, 100, "mini batch size per iteration.");
DEFINE_int32(coord_block, 30, "coordinates block size per iteration.");
DEFINE_int32(max_epoch, 400, "maximum epochs.");
DEFINE_int32(inner_iter, 1000, "inner iterations per epoch");
DEFINE_double(learning_rate, 1e-4, "Learning rate.");
DEFINE_double(first_epoch_learning_rate, 1e-4, "First epoch learning rate for svrg.");
DEFINE_double(learning_rate_dec, 0, "Learning rate decay. 1/(1+epoch)^eta");
DEFINE_double(weight_decay, 1e-4, "weight decay w^2 ");
DEFINE_double(epsilon, 1e-8, "small value to compute numerical gradient.");
DEFINE_string(data_dir, "/home/jonny/ZHOU/Data/lecun_usps/", "data directory");


#endif
