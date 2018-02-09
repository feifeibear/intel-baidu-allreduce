//#define HUGE

#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <sys/mman.h>

#include "collectives.h"
#include "timer.h"


struct MPIGlobalState {
    // Whether the global state (and MPI) has been initialized.
    bool initialized = false;
  std::vector<MPI_Comm> dups;
  int comm_threads;
};

// MPI relies on global state for most of its internal operations, so we cannot
// design a library that avoids global state. Instead, we centralize it in this
// single global struct.
static MPIGlobalState global_state;

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
//
// An exception is thrown if MPI or CUDA cannot be initialized.
void InitCollectives(int threads) {
  // CPU-only initialization.
  int provided,mpi_error;

  mpi_error = MPI_Init_thread(NULL,NULL,MPI_THREAD_MULTIPLE,&provided);
  assert (provided == MPI_THREAD_MULTIPLE);

  global_state.comm_threads=threads;
  global_state.dups.resize(threads);
  for(int t=0;t<global_state.comm_threads;t++){
    MPI_Comm_dup(MPI_COMM_WORLD,&global_state.dups[t]);
  }
  if(mpi_error != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Init failed with an error");
  }
  global_state.initialized = true;
}

std::map<float *,size_t> allocations;

// Allocate a new memory buffer on CPU or GPU.
float* alloc(size_t size) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#ifdef HUGE
  size_t org=size;
  uint64_t TwoMB = 2*1024*1024;

  size = size * sizeof(float);
  size = (size + (TwoMB ) ) & (~(TwoMB-1)); // Align up to page   boundary


 // CPU memory allocation through standard allocator.
  float *buf =(float *) mmap(NULL, size, PROT_READ | PROT_WRITE,  MAP_HUGETLB|MAP_SHARED| MAP_ANONYMOUS, -1, 0); 

  //  if (mpi_rank==0)  std::cout << "mmap "<<org <<" "<<size<<" "<< std::hex<<(uint64_t)buf<<std::dec<<std::endl;

  assert (buf!=MAP_FAILED);
  bzero(buf,size);

  allocations[buf]=size;
#else 
  float *buf = new float[size];
#endif 
  return buf;
}

// Deallocate an allocated memory buffer.
void dealloc(float* buffer) {
#ifdef HUGE
  size_t size = allocations[buffer];
  munmap(buffer,size);
  allocations.erase(buffer);
#else 
  delete[] buffer;
#endif
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(float* dst, float* src, size_t size) {
  // CPU memory allocation through standard allocator.
#pragma omp parallel for
  for(size_t i = 0; i < size; i++) {
    dst[i]=src[i];
  }
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(float* dst, float* src, size_t size) {
  // Accumulate values from `src` into `dst` on the CPU.
#pragma omp parallel for
  for(size_t i = 0; i < size; i++) {
    dst[i] += src[i];
  }
}

static void GetWork(int nwork, int me, int & mywork, int & myoff,int units){
  int basework = nwork/units;
  int backfill = units-(nwork%units);
  if ( me >= units ) { 
    mywork = myoff = 0;
  } else { 
    mywork = (nwork+me)/units;
    myoff  = basework * me;
    if ( me > backfill ) 
	myoff+= (me-backfill);
  }
  return;
};

static uint64_t bytes;
static double   elapsed;
static double   elapsed_all;

static float *buffer;
static float *output;
static size_t allocated_length; 
void Alloc(size_t length)
{
  if ( length == 0 ) length=1;
  if ( length > allocated_length ) {
    if ( allocated_length > 0 ) { 
      Dealloc();
    } 
    buffer = alloc(length);
    output = alloc(length);
    allocated_length=length;
  }
}
void Dealloc(void)
{
  dealloc(buffer);
  dealloc(output);
  allocated_length=0;
}

void Report(void) {
  std::cout << bytes << " bytes in " <<elapsed << " s "<< bytes/elapsed/1000/1000/1000 << " GB/s with threads = "
	    << global_state.comm_threads<<std::endl;

  std::cout << "Total time " <<elapsed_all << " s "<<std::endl;
}
void ResetCounters(void) {
  bytes=0;
  elapsed=0;
  elapsed_all=0;
}

void OMP_Sendrecv_float(const float *sendbuf, int sendcount, int dest  , int sendtag,
  		             float *recvbuf, int recvcount, int source, int recvtag)
{
  timer::Timer timer;
  timer.start();
#pragma omp parallel 
  {
    int t = omp_get_thread_num();
    if ( t < global_state.comm_threads ) {
      MPI_Comm comm = global_state.dups[t];
      int swork,rwork;
      int soff,roff;
      GetWork(sendcount,t,swork,soff,global_state.comm_threads);
      GetWork(recvcount,t,rwork,roff,global_state.comm_threads);

      MPI_Sendrecv((const void *)&sendbuf[soff],swork,MPI_FLOAT, dest  , sendtag,
		   (void *)      &recvbuf[roff],rwork,MPI_FLOAT, source, recvtag,
		   comm, MPI_STATUS_IGNORE);
    }
  }
  elapsed+=timer.seconds();
  bytes+=sendcount * sizeof(float);
  bytes+=recvcount * sizeof(float);
}

// Collect the input buffer sizes from all ranks using standard MPI collectives.
// These collectives are not as efficient as the ring collectives, but they
// transmit a very small amount of data, so that is OK.
std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length) {
    std::vector<size_t> lengths(size);
    MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
                  &lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    return lengths;
}



/* Perform a ring allreduce on the data. The lengths of the data chunks passed
 * to this function must be the same across all MPI processes. The output
 * memory will be allocated and written into `output`.
 *
 * Assumes that all MPI processes are doing an allreduce of the same data,
 * with the same size.
 *
 * A ring allreduce is a bandwidth-optimal way to do an allreduce. To do the allreduce,
 * the nodes involved are arranged in a ring:
 *
 *                   .--0--.
 *                  /       \
 *                 3         1
 *                  \       /
 *                   *--2--*
 *
 *  Each node always sends to the next clockwise node in the ring, and receives
 *  from the previous one.
 *
 *  The allreduce is done in two parts: a scatter-reduce and an allgather. In
 *  the scatter reduce, a reduction is done, so that each node ends up with a
 *  chunk of the final output tensor which has contributions from all other
 *  nodes.  In the allgather, those chunks are distributed among all the nodes,
 *  so that all nodes have the entire output tensor.
 *
 *  Both of these operations are done by dividing the input tensor into N
 *  evenly sized chunks (where N is the number of nodes in the ring).
 *
 *  The scatter-reduce is done in N-1 steps. In the ith step, node j will send
 *  the (j - i)th chunk and receive the (j - i - 1)th chunk, adding it in to
 *  its existing data for that chunk. For example, in the first iteration with
 *  the ring depicted above, you will have the following transfers:
 *
 *      Segment 0:  Node 0 --> Node 1
 *      Segment 1:  Node 1 --> Node 2
 *      Segment 2:  Node 2 --> Node 3
 *      Segment 3:  Node 3 --> Node 0
 *
 *  In the second iteration, you'll have the following transfers:
 *
 *      Segment 0:  Node 1 --> Node 2
 *      Segment 1:  Node 2 --> Node 3
 *      Segment 2:  Node 3 --> Node 0
 *      Segment 3:  Node 0 --> Node 1
 *
 *  After this iteration, Node 2 has 3 of the four contributions to Segment 0.
 *  The last iteration has the following transfers:
 *
 *      Segment 0:  Node 2 --> Node 3
 *      Segment 1:  Node 3 --> Node 0
 *      Segment 2:  Node 0 --> Node 1
 *      Segment 3:  Node 1 --> Node 2
 *
 *  After this iteration, Node 3 has the fully accumulated Segment 0; Node 0
 *  has the fully accumulated Segment 1; and so on. The scatter-reduce is complete.
 *
 *  Next, the allgather distributes these fully accumululated chunks across all nodes.
 *  Communication proceeds in the same ring, once again in N-1 steps. At the ith step,
 *  node j will send chunk (j - i + 1) and receive chunk (j - i). For example, at the
 *  first iteration, the following transfers will occur:
 *
 *      Segment 0:  Node 3 --> Node 0
 *      Segment 1:  Node 0 --> Node 1
 *      Segment 2:  Node 1 --> Node 2
 *      Segment 3:  Node 2 --> Node 3
 *
 * After the first iteration, Node 0 will have a fully accumulated Segment 0
 * (from Node 3) and Segment 1. In the next iteration, Node 0 will send its
 * just-received Segment 0 onward to Node 1, and receive Segment 3 from Node 3.
 * After this has continued for N - 1 iterations, all nodes will have a the fully
 * accumulated tensor.
 *
 * Each node will do (N-1) sends for the scatter-reduce and (N-1) sends for the allgather.
 * Each send will contain K / N bytes, if there are K bytes in the original tensor on every node.
 * Thus, each node sends and receives 2K(N - 1)/N bytes of data, and the performance of the allreduce
 * (assuming no latency in connections) is constrained by the slowest interconnect between the nodes.
 *
 */
void RingAllreduce(float* data, size_t length, float** output_ptr) {
  timer::Timer timer;
  timer.start();
  Alloc(length);

    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Check that the lengths given to every process are the same.
    std::vector<size_t> lengths = AllgatherInputLengths(size, length);
    for(size_t other_length : lengths) {
        if(length != other_length) {
            throw std::runtime_error("RingAllreduce received different lengths");
        }
    }

    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.
    const size_t segment_size = length / size;
    std::vector<size_t> segment_sizes(size, segment_size);

    const size_t residual = length % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // The last segment should end at the very end of the buffer.
    assert(segment_ends[size - 1] == length);

    // Allocate the output buffer.
    //    float* output = alloc(length);
    assert(length <= allocated_length);
     *output_ptr =  output;

    // Copy your data to the output buffer to avoid modifying the input buffer.
    copy(output, data, length);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.
    //    float* buffer = alloc(segment_sizes[0]);
    assert(segment_sizes[0] <= allocated_length);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    MPI_Status recv_status;
    MPI_Request recv_req;
    MPI_Datatype datatype = MPI_FLOAT;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).

    for (int i = 0; i < size - 1; i++) {
        int recv_chunk = (rank - i - 1 + size) % size;
        int send_chunk = (rank - i + size) % size;
        float* segment_send = &(output[segment_ends[send_chunk] -
				       segment_sizes[send_chunk]]);

	int tag = 0;
	OMP_Sendrecv_float(segment_send,segment_sizes[send_chunk],send_to  ,tag,
			   buffer      ,segment_sizes[recv_chunk],recv_from,tag);

        float *segment_update = &(output[segment_ends[recv_chunk] -
                                         segment_sizes[recv_chunk]]);

        reduce(segment_update, buffer, segment_sizes[recv_chunk]);
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).

    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + 1 + size) % size;
        int recv_chunk = (rank - i + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        float* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        float* segment_recv = &(output[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        OMP_Sendrecv_float(segment_send, segment_sizes[send_chunk], send_to  , 0, 
			   segment_recv, segment_sizes[recv_chunk], recv_from, 0);
    }
    elapsed_all+=timer.seconds();

    // Free temporary memory.
    //dealloc(buffer);
}

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
//
// For more information on the ring allgather, read the documentation for the
// ring allreduce, which includes a ring allgather as the second stage.
void RingAllgather(float* data, size_t length, float** output_ptr) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Get the lengths of data provided to every process, so that we know how
    // much memory to allocate for the output buffer.
    std::vector<size_t> segment_sizes = AllgatherInputLengths(size, length);
    size_t total_length = 0;
    for(size_t other_length : segment_sizes) {
        total_length += other_length;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    assert(segment_sizes[rank] == length);
    assert(segment_ends[size - 1] == total_length);

    // Allocate the output buffer and copy the input buffer to the right place
    // in the output buffer.
    //    float* output = alloc(total_length);
    *output_ptr = output;

    copy(output + segment_ends[rank] - segment_sizes[rank],
         data, segment_sizes[rank]);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    // What type of data is being sent
    MPI_Datatype datatype = MPI_FLOAT;

    MPI_Status recv_status;

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).

    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + size) % size;
        int recv_chunk = (rank - i - 1 + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        float* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        float* segment_recv = &(output[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        //MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
        //        datatype, send_to, 0, segment_recv,
        //        segment_sizes[recv_chunk], datatype, recv_from,
        //        0, MPI_COMM_WORLD, &recv_status);
        OMP_Sendrecv_float(segment_send, segment_sizes[send_chunk], send_to  , 0, 
		           segment_recv, segment_sizes[recv_chunk], recv_from, 0);
    }
}
