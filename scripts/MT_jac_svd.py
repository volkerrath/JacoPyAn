#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
# ---

"""
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules

import os
import sys

# import struct
import time
from datetime import datetime
import warnings
import gc


import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scs
import netCDF4 as nc




JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
import util as utl

from version import versionstrg

gc.enable()


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

rng = np.random.default_rng()
nan = np.nan

# KRAFLA case
# WorkDir = "/media/vrath/BlackOne/MT_Data/Krafla/Krafla1/"
# MFile   = WorkDir +r"Krafla.rho"

# Annecy case
WorkDir = "/home/vrath/MT_Data/Annecy/Jacobians/"

# if not os.path.isdir(WorkDir):
#     print("File: %s does not exist, but will be created" % WorkDir)
#     os.mkdir(WorkDir)
MFile = WorkDir+"ANN_best"
JFile = WorkDir +"ANN_ZPT_nerr_sp-8"


JThresh  = 1.e-4
# NumSingular = [ 100, 200, 300, 400, 500, 1000]
NumSingular = [ 500]
OverSample=  [2]
SubspaceIt = [0, 1, 2, 4]



total = 0.0
start =time.perf_counter()
print("\nReading Data from "+JFile)

Jac = scs.load_npz(JFile +"_jac.npz")
Dat = np.load(JFile +"_info.npz", allow_pickle=True)

Freq = Dat["Freq"]
Comp = Dat["Comp"]
Site = Dat["Site"]
DTyp = Dat["DTyp"]
Data = Dat["Data"]
Scale = Dat["Scale"]
Info = Dat["Info"]

elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile))

nsingval = NumSingular[0]
noversmp = OverSample[0]
nsubspit = SubspaceIt[0]

info = []
for noversmp in OverSample:
    for nsubspit in SubspaceIt:
        for nsingval in NumSingular:
            start = time.perf_counter()
            U, S, Vt = jac.rsvd(Jac.T,
                                rank=nsingval,
                                n_oversamples=noversmp*nsingval,
                                n_subspace_iters=nsubspit)
            elapsed = time.perf_counter() - start
            print("Used %7.4f s for calculating k = %i SVD " % (elapsed, nsingval))
            print("Oversamplinng factor =  ", str(noversmp))
            print("Subspace iterations  =  ", str(nsubspit))

            D = U@scs.diags(S[:])@Vt - Jac.T
            x_op = np.random.normal(size=np.shape(D)[1])
            n_op = npl.norm(D@x_op)/npl.norm(x_op)
            j_op = npl.norm(Jac.T@x_op)/npl.norm(x_op)
            perc = 100. - n_op*100./j_op
            tmp = [nsingval, noversmp, nsubspit, perc, elapsed]
            info.append(tmp)

            print(" Op-norm J_k = "+str(n_op)+", explains "
                +str(perc)+"% of variations")
            print("")

            File = JFile+"_SVD_k"+str(nsingval)\
                    +"_o"+str(noversmp)\
                    +"_s"+str(nsubspit)\
                    +"_"+str(np.around(perc,1))\
                    +"%.npz"

            np.savez_compressed(File, U=U, S=S, V=Vt, Nop=perc)

np.savetxt(JFile+"_run.dat",  np.vstack(info),
                fmt="%6i, %6i, %6i, %4.6g, %4.6g")
