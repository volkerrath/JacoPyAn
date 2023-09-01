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
from sys import exit as error


import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scp
import netCDF4 as nc


import vtk
import pyvista as pv
import PVGeo as pvg

JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


rng = np.random.default_rng()
nan = np.nan

normalize_err = True

normalize_max = False

sparsify = False
sparse_thresh = 1.e-7

WorkDir = r"/home/vrath/work/MT_Data/Ubaye/UB22_jac_best/"
MFile   = WorkDir +r"Ub22_ZoffPT_02_NLCG_014.rho"
JFile = [WorkDir+r"Ub22_Zoff.jac", WorkDir+r"Ub22_T.jac", WorkDir+r"Ub22_P.jac", ]
DFile = [WorkDir+r"Ub22_Zoff.dat", WorkDir+r"Ub22_T.dat", WorkDir+r"Ub22_P.dat", ]



total = 0.0


start = time.time()
dx, dy, dz, rho, reference = mod.read_model_mod(MFile, trans="log10")
dims = np.shape(rho)
# print(dims)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


if np.size(DFile) != np.size(JFile):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFile)


mxVal = 1e-30
mxLst = []
for f in np.arange(nF):

    name, ext = os.path.splitext(JFile[f])
    start =time.time()
    print("\nReading Data from "+DFile[f])
    Data, Site, Freq, Comp, Head = mod.read_data_jac(DFile[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFile[f]))
    total = total + elapsed

    start = time.time()
    print("Reading Jacobian from "+JFile[f])
    Jac = mod.read_jac(JFile[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile[f]))
    total = total + elapsed

    nstr = ""
    if normalize_err:
        nstr = nstr+"_nerr"
        start = time.time()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 7], (dsh[0], 1))
        mx0 = np.max(np.abs(Jac))
        Jac = jac.normalize_jac(Jac, err)
        elapsed = time.time() - start
        print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFile[f]))

    mx = np.max(np.abs(Jac))
    mxLst.append(mx)
    mxVal = np.amax([mxVal,mx])

    if normalize_max:
        nstr = nstr+"_max"
        start = time.time()
        Jac = jac.normalize_jac(Jac,[mx])
        elapsed = time.time() - start
        total = total + elapsed
        print(" Max value is %7.4f, before was %7.4f" % (mx, mx0))


    sstr=""
    if sparsify:
        sstr="_sp"+str(round(np.log10(sparse_thresh)))
        start = time.time()
        Jacs= jac.sparsify_jac(Jac,sparse_thresh=sparse_thresh)
        elapsed = time.time() - start
        total = total + elapsed
        print(" Used %7.4f s for sparsifying Jacobian %s " % (elapsed, JFile[f]))
        NPZFile = name+nstr+sstr+".npz"
        np.savez_compressed(NPZFile,
                            Jac=Jacs, Data=Data, Site=Site, Comp=Comp)
        elapsed = time.time() - start
        total = total + elapsed
        print(" Used %7.4f s for writing sparsified Jacobian to %s " % (elapsed, NPZFile))


    start = time.time()
    S, Smax = jac.calculate_sens(Jac, normalize=False, small=1.0e-14)
    S = np.reshape(S, dims, order="F")
    elapsed = time.time() - start
    total = total + elapsed
    print(" Used %7.4f s for calculating Sensitivity from Jacobian  %s " % (elapsed, JFile[f]))

    start = time.time()
    SNSFile = name+nstr+".sns"
    mod.write_model_mod(SNSFile, dx, dy, dz, S, reference, trans="log10")
    elapsed = time.time() - start
    total = total + elapsed
    print(" Used %7.4f s for writing Sensitivity from Jacobian  %s " % (elapsed, JFile[f]))

    start = time.time()
    NCFile = name + nstr+".nc"
    mod.write_jac_ncd(NCFile, Jac, Data, Site, Comp)
    elapsed = time.time() - start
    total = total + elapsed
    print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NCFile))

    start = time.time()
    NPZFile = name +nstr+ ".npz"
    np.savez_compressed(NPZFile,
                        Jac=Jac, Data=Data, Site=Site, Comp=Comp)
    elapsed = time.time() - start
    total = total + elapsed
    print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NPZFile))
