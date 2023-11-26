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


SparseThresh = 1.e-7
Sparse = SparseThresh > 0

ErrorScale = False
VolumeScale= False

Scale = 1.


WorkDir = JACOPYAN_DATA+"/Sabancaya/"
WorkDir = JACOPYAN_DATA+"/Peru/Sabancaya//SABA8_Jac/"

if not WorkDir.endswith("/"):
    WorkDir = WorkDir+"/"
MFile = WorkDir + "SABA8_best.rho"

# JFiles = [WorkDir+"SABA8_Z.jac", WorkDir+"SABA8_P.jac", WorkDir+"SABA8_T.jac",]
# DFiles = [WorkDir+"SABA8_Z_jac.dat", WorkDir +
#           "SABA8_P_jac.dat", WorkDir+"SABA8_T_jac.dat",]

JFiles = [WorkDir+"SABA8_Pi.jac",]
DFiles = [WorkDir+"SABA8_Pi_jac.dat",]

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
blank = 1.e-30 #np.nan

elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


name, ext = os.path.splitext(MFile)


TSTFile = name 
Head = "#  original model"
mod.write_mod(TSTFile, ModExt="_0_MaskTest.rho", trans="LOGE",
                  dx=dx, dy=dy, dz=dz, mval=rho,
                  reference=reference, mvalair=rhoair, aircells=aircells, header=Head)

airmask = jac.set_airmask(rho=rho, aircells=aircells, flat = False, out=True)

rhotest = airmask.reshape(dims)*rho
mod.write_mod(TSTFile, ModExt="_1_MaskTest.rho", trans="LOGE",
                  dx=dx, dy=dy, dz=dz, mval=rhotest,
                  reference=reference, mvalair=rhoair, aircells=aircells, header=Head)


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
    

    # mx = np.max(Jac)
    # mn = np.amin(Jac)
    # print(JFiles[f]+" minimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))
    # airmask = airmask.flatten(order="F")
    wcell   = 1. /vcell.flatten(order="F")

    
    # GG = Jac[1000,:]
    # OO = np.ones_like(GG)
    # snstest = airmask.reshape(dims, order = "F")*GG.reshape(dims, order = "F")
    # mod.write_mod(TSTFile, ModExt="_1_MaskTest.sns", trans="LINEAR",
    #               dx=dx, dy=dy, dz=dz, mval=snstest,
    #               reference=reference, mvalair=rhoair, aircells=aircells, header=Head)
    
    
    if np.shape(Jac)[0]!=np.shape(Data)[0]:
        print(np.shape(Jac),np.shape(Data))
        error(" Dimensions of Jacobian and data do not match! Exit.")

    # if ErrorScale:
    #     nstr = nstr+"_nerr"
    #     start = time.perf_counter()
    #     dsh = np.shape(Data)
    #     err = np.reshape(Data[:, 5], (dsh[0], 1))
    #     print(np.amin(err), np.amax(err))
    #     Jac, _ = jac.normalize_jac(Jac, err)
    #     elapsed = time.perf_counter() - start
    #     print(" Used %7.4f s for normalizing Jacobian with data error from %s " %
    #           (elapsed, DFiles[f]))
    #     start = time.perf_counter()

    # if VolumeScale:
        
    #     nstr = nstr+"_vcell"
    #     start = time.perf_counter()
    #     wcell   = 1. /vcell.flatten(order="F")
    #     print(np.amin(err), np.amax(err))
    #     Jac = Jac*wcell
    #     elapsed = time.perf_counter() - start
    #     print(" Used %7.4f s for normalizing Jacobian with inverse cell volumes " %
    #           (elapsed))
    #     start = time.perf_counter()

    sstr = "_full"
    if SparseThresh > 0.:
        # airmask = airmask.flatten(order="F")
        sstr = "_sp"+str(round(np.log10(SparseThresh)))
        start = time.perf_counter()
        for idt in np.arange(np.shape(Jac)[0]):
            JJ = Jac[idt,:].reshape(dims,order="F")
            Jac[idt,:] = (airmask*JJ).flatten(order="F")
        
        mx = np.nanmax(Jac)
        mn = np.nanmin(Jac)
        print(JFiles[f]+" minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx)) 
        
        Scale = np.nanmax(np.abs(Jac))
        Jac, Scale = jac.sparsify_jac(Jac, scalval=Scale, sparse_thresh=SparseThresh)
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
