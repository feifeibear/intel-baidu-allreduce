# Check that MPI path exists.
# Check that CUDA path exists.

CXX=mpiCC #$(HOME)/soft_install/bin/mpicxx #mpicxx #mpiicpc
#LDFLAGS:=-fopenmp
CFLAGS:=-O2 -I.
EXE_NAME:=allreduce-test
SRC:=$(wildcard *.cpp test/*.cpp)
OBJS:=$(SRC:.cpp=.o) sw_add.o sw_slave_add.o

all: $(EXE_NAME)

$(EXE_NAME): $(OBJS)
	$(CXX) -o $(EXE_NAME) $^ $(LDFLAGS) 

%.o: %.cpp
	$(CXX) -host -c $(CFLAGS) $< -o $@

sw_add.o:sw_add.c
	sw5cc.new -host -c -msimd -O2 sw_add.c -o sw_add.o

sw_slave_add.o: ./slave/sw_slave_add.c
	sw5cc.new -slave -c -msimd -O2 $^ -o $@ 

test: $(EXE_NAME)
	$(EXE_NAME)

clean:
	rm -f *.o test/*.o $(EXE_NAME)
