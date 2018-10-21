set -x
bsub -b -I -p -q q_sw_share -N 4 -share_size 7000 -o run.log -cgsp 64 ./allreduce-test
#bsub -b -I -p -q q_sw_expr -N 4 -sw3run ./sw3run-all -sw3runarg "-a 1" -sw3runarg -cross_size 28000 -o run.log -cgsp 64 ./allreduce-test
