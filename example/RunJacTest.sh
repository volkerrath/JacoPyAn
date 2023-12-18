#! /bin/bash

# init_intel compiler
. /opt/intel/oneapi/setvars.sh intel64 ilp64

mpirun -np 10  /home/vrath/bin/iMod3DMTJ.x -J JacTest.rho JacTest_Z.dat JacTest_Z.jac  JacTest.fwd  JacTest.cov > JacTest_Z.out


