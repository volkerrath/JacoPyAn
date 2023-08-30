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


mypath = ["/home/vrath/Py4MT/JacoPyAn/modules/", "/home/vrath/Py4MT/JacoPyAn/scripts/"]
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


Tasks = ["write_npz", "normalize_err", "sparsify"]
"""
"normalize_err", "sparsify". "split", "merge"
"""
sparse_thresh = 1.e-8


outform = "LINEAR"
outform = outform.upper()

WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/UbiJac/"
WorkName = "UBI_best"
MFile   = WorkDir + "UBI_best.rho"
MPad=[14, 14 , 14, 14, 0, 71]

JFiles = [WorkDir+"UBI_best.jac", ]
DFiles = [WorkDir+"UBI_best_jac.dat", ]

if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


total = 0.0
start = time.time()
dx, dy, dz, rho, reference, _, vcell = mod.read_model(MFile, trans="linear", volumes=True)
dims = np.shape(rho)
sdims = np.size(rho)

rhoair = 1.e17
aircells = np.where(rho>rhoair/10)


# TSTFile = WorkDir+WorkName+"0_MaskTest.rho"
# mod.write_model_mod(TSTFile, dx, dy, dz, rho, reference, trans="LINEAR", mvalair=blank, aircells=aircells)


jacmask = jac.set_mask(rho=rho, pad=MPad, blank= blank, flat = False, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)

# rhotest = jacmask.reshape(dims)*rho
# TSTFile = WorkDir+WorkName+"1_MaskTest.rho"
# mod.write_model_mod(TSTFile, dx, dy, dz, rhotest, reference, trans="LINEAR", mvalair=blank, aircells=aircells)

elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


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
    if write_npz:
        start = time.time()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 5], (dsh[0], 1))
        print(np.amin(err), np.amax(err))
        Jac = jac.normalize_jac(Jac, err)
        NPZFile = name +".npz"
        np.savez_compressed(NPZFile,
                            Jac=Jac, Data=Data, Site=Site, Comp=Comp)
        elapsed = time.time() - start
        total = total + elapsed
        print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NPZFile))

    nstr = ""
    if normalize_err:
        nstr = nstr+"_nerr"
        start = time.time()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 5], (dsh[0], 1))
        print(np.amin(err), np.amax(err))
        Jac = jac.normalize_jac(Jac, err)
        elapsed = time.time() - start
        print(" Used %7.4f s for normalizing Jacobian with data error from %s " % (elapsed, DFile))
        start = time.time()
        NPZFile = name +nstr+ ".npz"
        np.savez_compressed(NPZFile,
                            Jac=Jac, Data=Data, Site=Site, Comp=Comp)
        elapsed = time.time() - start
        total = total + elapsed
        print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NPZFile))
    
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
