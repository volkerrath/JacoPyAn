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
        sys.path.insert(0, pth)

import util as utl
from version import versionstrg
import modem as mod
import jacproc as jac

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


rng = np.random.default_rng()
nan = np.nan


SparseThresh = 1.e-8
Sparse = SparseThresh > 0

Scale = 1.


WorkDir = JACOPYAN_ROOT+"/work/SABA/"
if not WorkDir.endswith("/"):
    WorkDir = WorkDir+"/"
MFile = WorkDir + "SABA8_best.rho"
# JFiles = [WorkDir+"SABA8_Z.jac", WorkDir+"SABA8_P.jac", WorkDir+"SABA8_T.jac",]
# DFiles = [WorkDir+"SABA8_Z_jac.dat", WorkDir +
#           "SABA8_P_jac.dat", WorkDir+"SABA8_T_jac.dat",]
JFiles = [WorkDir+"SABA8_Z.jac",]
DFiles = [WorkDir+"SABA8_Z_jac.dat",]

# WorkDir = JACOPYAN_ROOT+"/work/TestJac/"
# if not WorkDir.endswith("/"):
#     WorkDir = WorkDir+"/"
# MFile = WorkDir + "TestJac.rho"
# MPad = [0, 0,    0,  0,    0, 0]
# JFiles = [WorkDir+"TestJac_Z.jac", WorkDir+"TestJac_P.jac", WorkDir+"TestJac_T.jac",]
# DFiles = [WorkDir+"TestJac_Z_jac.dat", WorkDir +
#           "TestJac_P_jac.dat", WorkDir+"TestJac_T_jac.dat",]

if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


total = 0.0
start = time.perf_counter()
dx, dy, dz, rho, reference, _, vcell = mod.read_mod(
    MFile, trans="linear", volumes=True)
dims = np.shape(rho)
sdims = np.size(rho)

rhoair = 1.e17
aircells = np.where(rho > rhoair/10)
blank = rhoair

elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


for f in np.arange(nF):
    nstr = ""
    name, ext = os.path.splitext(JFiles[f])
    start = time.perf_counter()
    print("\nReading Data from "+DFiles[f])
    Data, Site, Freq, Comp, DTyp, Head = mod.read_data_jac(DFiles[f])
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFiles[f]))
    total = total + elapsed
    print(np.unique(DTyp))
    start = time.perf_counter()
    print("Reading Jacobian from "+JFiles[f])
    Jac, Info = mod.read_jac(JFiles[f])
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
    total = total + elapsed

    if np.shape(Jac)[0]!=np.shape(Data)[0]:
        print(np.shape(Jac),np.shape(Data))
        error(" Dimensions of Jacobian and data do not match! Exit.")

    nstr = nstr+"_nerr"
    start = time.perf_counter()
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    print(np.amin(err), np.amax(err))
    Jac = jac.normalize_jac(Jac, err)
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for normalizing Jacobian with data error from %s " %
          (elapsed, DFiles[f]))
    start = time.perf_counter()

    sstr = "_full"
    if SparseThresh > 0.:

        sstr = "_sp"+str(round(np.log10(SparseThresh)))
        start = time.perf_counter()
        Jac, Scale = jac.sparsify_jac(Jac, sparse_thresh=SparseThresh)
        elapsed = time.perf_counter() - start
        total = total + elapsed
        print(" Used %7.4f s for sparsifying Jacobian %s " %
              (elapsed, JFiles[f]))

    name = name+nstr+sstr
    start = time.perf_counter()

    np.savez_compressed(name + "_info.npz", Freq=Freq, Data=Data, Site=Site, Comp=Comp,
                        Info=Info, DTyp=DTyp, Scale=Scale, allow_pickle=True)
    if Sparse:
        scs.save_npz(name + "_jac.npz", matrix=Jac)  # , compressed=True)
    else:
        np.savez_compressed(name +"_jac.npz", Jac)
    elapsed = time.perf_counter() - start
    total = total + elapsed
    print(" Used %7.4f s for writing Jacobian and infp to %s " % (elapsed, name))
