#! /bin/bash

# init_intel compiler
. /opt/intel/oneapi/setvars.sh intel64 ilp64

mpirun -np 10  /home/vrath/bin/iMod3DMTJ.x -J JacTest.rho JacTest_Z.dat JacTest_Z.jac  JacTest.fwd  JacTest.cov > JacTest_Z.out

mpirun -np 10  /home/vrath/bin/iMod3DMTJ.x -J JacTest.rho JacTest_T.dat JacTest_T.jac  JacTest.fwd  JacTest.cov > JacTest_T.out

mpirun -np 10  /home/vrath/bin/iMod3DMTJ.x -J JacTest.rho JacTest_P.dat JacTest_P.jac  JacTest.fwd  JacTest.cov > JacTest_P.out
