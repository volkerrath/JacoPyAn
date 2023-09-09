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


SparseThresh = 1.e-8

Task = "merge"


WorkDir = JACOPYAN_ROOT+"/work/"
#WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/UbiJacNewFormat/"
WorkName = "UBI_ZPT"

JFiles = [WorkDir+"//UBI_Z.jac", WorkDir+"//UBI_P.jac",WorkDir+"//UBI_T.jac",]
DFiles = [WorkDir+"//UBI_Z_jac.dat", WorkDir+"//UBI_P_jac.dat",WorkDir+"//UBI_T_jac.dat",]

if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)

total = 0.

if "merg" in Task.lower():
    
    
    for ifile in np.arange(nF):
        
        name, ext = os.path.splitext(JFiles[ifile])
        start =time.time()
        print("\nReading Data from "+DFiles[ifile])
        Data, Site, Freq, Comp, Dtype, Head = mod.read_data_jac(DFiles[ifile])
        elapsed = time.time() - start
        print(" Used %7.4f s for reading Data from %s " % (elapsed, DFiles[ifile]))
        total = total + elapsed
    
        start = time.time()
        print("Reading Jacobian from "+JFiles[ifile])
        Jac = mod.read_jac(JFiles[ifile])
        elapsed = time.time() - start
        print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[ifile]))
        total = total + elapsed
        
        if ifile==0:
            Jac_merged = Jac 
            Data_merged = Data 
            Site_merged = Site
            Freq_mergerd = Freq
            Comp_merged = Comp
            Dtype_merged = Dtype 
        else:
            Jac_merged = scs.hstack((Jac_merged, Jac))
           #  Data_merged = Data 
           # Site_merged = Site
           # Freq_mergerd = Freq
           # Comp_merged = Comp
           # Dtype_merged = Dtype 
          
            
            
 
    # if SparseThresh > 0.:
    #     sstr="_sp"+str(round(np.log10(SparseThresh)))
    #     start = time.time()
    #     Jacs= jac.sparsify_jac(Jac,SparseThresh=SparseThresh)
    #     elapsed = time.time() - start
    #     total = total + elapsed
    # else:     
    #     sstr = "_full"
    #     if SparseThresh>0.:
    
    #         sstr="_sp"+str(round(np.log10(SparseThresh)))
    #         start = time.time()
    #         Jac, _= jac.sparsify_jac(Jac, SparseThresh=SparseThresh)
    #         elapsed = time.time() - start
    #         total = total + elapsed
    #         print(" Used %7.4f s for sparsifying Jacobian %s " % (elapsed, JFiles[ifile]))
    
    
    # name = name+
    # start = time.time()
    # NPZFile = name +"_info.npz"
    # np.savez_compressed(NPZFile, Freq=Freq, Data=Data, Site=Site, Comp=Comp, Info=Info, Dtype=Dtype, allow_pickle=True)
    # NPZFile = name +"_jac.npz"
    # if SparseThresh>0.:
    #    sps.save_npz(NPZFile, matrix=Jac, compressed=True)
    # else: 
    #    np.savez_compressed(NPZFile, Jac)
    # elapsed = time.time() - start
    # total = total + elapsed
    print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NPZFile))

  
