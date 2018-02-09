# Check that MPI path exists.
# Check that CUDA path exists.

CXX=$(HOME)/soft_install/bin/mpicxx #mpicxx #mpiicpc
LDFLAGS:=-fopenmp
CFLAGS:=-std=c++11  -I. -fopenmp
EXE_NAME:=allreduce-test
SRC:=$(wildcard *.cpp test/*.cpp)
OBJS:=$(SRC:.cpp=.o) 

all: $(EXE_NAME)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

$(EXE_NAME): $(OBJS)
	$(CXX) -o $(EXE_NAME) $^ $(LDFLAGS) 

test: $(EXE_NAME)
	$(EXE_NAME)

clean:
	rm -f *.o test/*.o $(EXE_NAME)
