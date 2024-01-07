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

# gc.enable()


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
MFile = WorkDir+"ANN_best.rho"
JFile = WorkDir +"ANN_ZPT_sp-8"


JThresh  = 1.e-4
NSingulr = 300

# NSamples = 10000
# NBodies  = 32
# x_bounds = [-3000., 3000.]
# y_bounds = [-3000., 3000.]
# z_bounds = [-300., 3000.]
# rad_bounds = [100.,1000.]
# res_bounds = [-0.3, 0.3]



total = 0.0

start = time.perf_counter()


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

# mu = 0.0
# sigma = 0.5
# r = rho.flat
# nproj = 1000

start = time.perf_counter()
U, S, Vt = Jac.rsvd(Jac.T, rank=NSingulr, n_oversamples=0, n_subspace_iters=0)
elapsed = time.perf_counter() - start
print(
    "Used %7.4f s for calculating k = %i SVD from %s " % (elapsed, NSingulr, JFile)
)

D = U@scs.diags(S[:])@Vt - Jac.T
x_op = np.random.normal(size=np.shape(D)[1])
n_op = npl.norm(D@x_op)/npl.norm(x_op)
j_op = npl.norm(Jac.T@x_op)/npl.norm(x_op)
print(" Op-norm J_k = "+str(n_op)+", explains "
      +str(100. - n_op*100./j_op)+"% of variations")


# m_avg = 0.
# v_avg = 0.
# s = time.perf_counter()
# for isample in np.arange(NSamples):

#     body = [
#     "ellipsoid", "add",
#     0., 0., 0.,
#     3000.,
#     1000., 2000., 1000.,
#     0., 0., 30.]

# # m = r + np.random.normal(mu, sigma, size=np.shape(r))
# #     t = time.perf_counter() - s
# #     print(" Used %7.4f s for generating m  " % (t))

# #     s = time.perf_counter()
# #     for proj in range(nproj):
# #         p = jac.projectMod(m, U)

# #     t = time.perf_counter() - s
# #     print(" Used %7.4f s for %i projections" % (t, nproj))

# # total = total + elapsed
# # print(" Total time used:  %f s " % (total))
# NSMFile = WorkDir+"Krafla1_Ellipsoids_median.sns"
# tmp = []
# tmp = np.reshape(tmp, dims, order="F")
# mod.write_model_mod(NSMFile, dx, dy, dz, S, reference, trans="linear", air=aircells)
# print(" Sensitivities written to "+NSMFile)
