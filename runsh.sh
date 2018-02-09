mkdir -p ./logs
export NUM_PROC=4
mpirun -np ${NUM_PROC} ./allreduce-test 4 > ./logs/${NUM_PROC}.log &
