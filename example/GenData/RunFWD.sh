#! /bin/bash

# This will only generate ZT data set, P/T/Z sets, and frequency choice need to be  done, e.e. with 3d-Grid.

# init_intel compiler
. /opt/intel/oneapi/setvars.sh intel64 ilp64

mpirun -np 10  /home/vrath/bin/iMod3DMTJ.x -F ObliqueOne.rho ZT_3sites_16periods.dat FWD_ObliqueOne_ZTnew.dat /dev/null FWD_para.dat
