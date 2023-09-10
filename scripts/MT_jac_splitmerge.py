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
import scipy.linalg as scl
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

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


WorkDir = JACOPYAN_ROOT+"/work/"
#WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/UbiJacNewFormat/"


Task = "merge"
MergedFile = "UBI_ZPT_sp-8"
MFiles = [WorkDir+"//UBI_Z_nerr_sp-8", WorkDir+"//UBI_P_nerr_sp-8",WorkDir+"//UBI_T_nerr_sp-8",]
nF = np.size(MFiles)

# Task = "split"
# SplitFile = "UBI_ZPT_sp-8"
# Split = ["freq"]



if "mer" in Task.lower():
    
    
    for ifile in np.arange(nF):     
        name, ext = os.path.splitext(MFiles[ifile])
        start =time.time()
        print("\nReading Data from "+MFiles[ifile])
        Jac = scs.load_npz(MFiles[ifile] +"_jac.npz")
        normalized = True
        
        tmp = np.load( MFiles[ifile] +"_info.npz", allow_pickle=True)
        Freqs = tmp["Freq"]
        Comps = tmp["Comp"]
        Sites = tmp["Site"]
        Dtype = tmp["Dtype"]
        Data = tmp["Data"]
        Scale = tmp["Scale"]
        Info = tmp["Info"]
        
        elapsed = time.time() - start
        print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, MFiles[ifile]))
        
        
       
        if ifile==0:
            Jac_merged = Jac 
            print(ifile, type(Jac_merged), type(Jac), np.shape(Jac_merged))
            Data_merged = Data 
            Site_merged = Sites
            Freq_merged = Freqs
            Comp_merged = Comps
            Dtyp_merged = Dtype 
            Infblk = Info
            Scales = Scale
        else:
            Jac_merged = scs.vstack((Jac_merged, Jac))            
            print(ifile, type(Jac_merged), np.shape(Jac_merged))
            Data_merged = np.vstack((Data_merged, Data)) 
            Site_merged = np.hstack((Site_merged, Sites))
            Freq_merged = np.hstack((Freq_merged, Freqs))
            Comp_merged = np.hstack((Comp_merged, Comps))
            Dtyp_merged = np.hstack((Dtyp_merged, Dtype))
            Infblk = np.vstack((Infblk, Info))
            Scales = np.hstack((Scales, Scale))
          
    # Scale = np.amax(Scales)
            

    start = time.time()
    np.savez_compressed(WorkDir+MergedFile+"_info.npz",
                        Freq=Freq_merged, Data=Data_merged, Site=Site_merged, 
                        Comp=Comp_merged, Info=Infblk, Dtype=Dtyp_merged, 
                        Scale=Scales, allow_pickle=True)
    scs.save_npz(WorkDir+MergedFile+"_jac.npz", matrix=Jac_merged, compressed=True)

    elapsed = time.time() - start
    print(" Used %7.4f s for storing Jacobian to %s " % (elapsed, WorkDir+MergedFile))

if "spl" in Task.lower():
    pass
