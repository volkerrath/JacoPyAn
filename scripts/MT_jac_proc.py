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
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


rng = np.random.default_rng()
nan = np.nan



SparseThresh = 1.e-8
Scale = 1.



WorkDir = JACOPYAN_ROOT+"/work/"
#WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/UbiJacNewFormat/"
WorkName = "UBI_ZPT"
MFile   = WorkDir + "UBI_best.rho"
MPad=[14, 14 , 14, 14, 0, 71]

# JFiles = [WorkDir+"//UBI_Z.jac", WorkDir+"//UBI_P.jac",WorkDir+"//UBI_T.jac",]
# DFiles = [WorkDir+"//UBI_Z_jac.dat", WorkDir+"//UBI_P_jac.dat",WorkDir+"//UBI_T_jac.dat",]

JFiles = [WorkDir+"//UBI_Z.jac"]
DFiles = [WorkDir+"//UBI_Z_jac.dat"]

if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


total = 0.0
start = time.time()
dx, dy, dz, rho, reference, _, vcell = mod.read_mod(MFile, trans="linear", volumes=True)
dims = np.shape(rho)
sdims = np.size(rho)

rhoair = 1.e17
aircells = np.where(rho>rhoair/10)
blank = rhoair

elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


for f in np.arange(nF):
    nstr = ""
    name, ext = os.path.splitext(JFiles[f])
    start =time.time()
    print("\nReading Data from "+DFiles[f])
    Data, Site, Freq, Comp, DTyp, Head = mod.read_data_jac(DFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFiles[f]))
    total = total + elapsed
    print(np.unique(DTyp))
    start = time.time()
    print("Reading Jacobian from "+JFiles[f])
    Jac, Info = mod.read_jac(JFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
    total = total + elapsed
    
    
    nstr = nstr+"_nerr"
    start = time.time()
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    print(np.amin(err), np.amax(err))
    Jac = jac.normalize_jac(Jac, err)
    elapsed = time.time() - start
    print(" Used %7.4f s for normalizing Jacobian with data error from %s " % (elapsed, DFiles[f]))
    start = time.time()
        
    sstr = "_full"
    if SparseThresh>0.:

        sstr="_sp"+str(round(np.log10(SparseThresh)))
        start = time.time()
        Jac, Scale = jac.sparsify_jac(Jac, sparse_thresh=SparseThresh)
        elapsed = time.time() - start
        total = total + elapsed
        print(" Used %7.4f s for sparsifying Jacobian %s " % (elapsed, JFiles[f]))
    
    
    name = name+nstr+sstr
    start = time.time()
    NPZFile = name +"_info.npz"
    np.savez_compressed(NPZFile, Freq=Freq, Data=Data, Site=Site, Comp=Comp, 
                        Info=Info, DTyp=DTyp, Scale=Scale, allow_pickle=True)
    NPZFile = name +"_jac.npz"
    if SparseThresh>0.:
       scs.save_npz(NPZFile, matrix=Jac) #, compressed=True)
    else: 
       np.savez_compressed(NPZFile, Jac)
    elapsed = time.time() - start
    total = total + elapsed
    print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NPZFile))

  

