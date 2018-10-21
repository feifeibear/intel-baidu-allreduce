#include "collectives.h"
#include <cstdlib>
#include "timer.h"
#include  <math.h>
#include <iomanip>
#include <mpi.h>
#include <stdexcept>
#include <iostream>
#include <vector>

int threads;
void TestCollectivesCPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        size_t size = sizes[i];
        int iters = iterations[i];

        float* data = (float*)malloc(size*sizeof(float));
        float* data_ref = (float*)malloc(size*sizeof(float));
        float* res_ref = (float*)malloc(size*sizeof(float));
        float seconds = 0.0f, mpi_seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                data[j] = data_ref[j] =  (float)rand()/RAND_MAX;
            }

            timer.start();
            MPI_Allreduce(data, data_ref, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            mpi_seconds += timer.seconds();

            timer.start();
            //RingAllreduce(data, size, iter == -1);
            TreeAllreduce(data, size, iter == -1);
            seconds += timer.seconds();


            // Check that we get the expected result.
            int error_cnt = 10;
            float sum1 = 0.0, sum2 = 0.0;
            for(size_t j = 0; j < size; j++) {
                if(fabs(data[j] - data_ref[j]) > 1e-3 && error_cnt) {
                    std::cerr << "error@: "<< j << ' ' << data[j] << ", " << data_ref[j] << std::endl;
                    error_cnt--;
                }
                sum1 += data[j];
                sum2 += data_ref[j];
            }
            if(error_cnt == 0)
              std::cout << "sum : " << sum1 << " sum_ref: " << sum2  << std::endl;
        }
        if(mpi_rank == 0) {
          /*
            std::cout << "size(KB) "
                << "\t" 
                << "time(sec) \t"
                << "MPI(GB/s) \t"
                << "Ring(GB/s)"
                << std::endl;
                */

            std::cout << size * sizeof(float)/1024 << ", "
                //<< mpi_seconds / iters << ", "
                << (float)size*sizeof(float)*2/(1024*1024*1024)/(mpi_seconds / iters)
                //<< ", "
                //<< seconds / iters
                << ", "
                <<  (float)size*sizeof(float)*2/(1024*1024*1024)/(seconds / iters)
                << std::endl;
	          //Report();
        }

        free(data);
        free(data_ref);
        free(res_ref);
    }
}

// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {
    //if(argc != 2) {
    //    std::cerr << "Usage: ./allreduce-test threads" << std::endl;
    //    return 1;
    //}
    //threads = atoi(argv[1]);
    MPI_Init(&argc, &argv);
#ifdef USE_SW
    thread_init();
#endif

    // Buffer sizes used for tests.
    std::vector<size_t> buffer_sizes; //= { 4096, 16384, 65536, 262144, 1048576, 8388608, 67108864 };
    std::vector<size_t> iterations;
    for(int i = 10; i <= 20; ++i) {
      buffer_sizes.push_back(128*pow(2,i));
      iterations.push_back(10);
    }

    // Number of iterations to run for each buffer size.
    //= {
    //    10, 10, 10, 10,
    //    100, 100, 100, 100,
    //    100, 100, 100    };

    // Test on either CPU and GPU.
    TestCollectivesCPU(buffer_sizes, iterations);

    // Finalize to avoid any MPI errors on shutdown.
    MPI_Finalize();

    return 0;
}
