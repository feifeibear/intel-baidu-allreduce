#include <mpi.h>
#include <iostream>

int main() {
  int provided;
  int mpi_error = MPI_Init_thread(NULL,NULL,MPI_THREAD_MULTIPLE,&provided);
  std::cout << "provided " << provided << std::endl;
  //int mpi_error = MPI_Init(NULL, NULL);
  if(mpi_error != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Init failed with an error");
  }
  int mpi_rank;
  if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
    throw std::runtime_error("get rank failed");
  std::cout << "Over " << mpi_rank << std::endl;
  MPI_Finalize();
  return 0;
}
