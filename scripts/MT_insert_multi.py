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
Reads ModEM model, reads ModEM"s Jacobian, does fancy things.

Created on Sun Jan 17 15:09:34 2021

@author: vrath jan 2021

"""

# Import required modules

import os
import sys
from sys import exit as error
# import struct
import time
from datetime import datetime

import numpy as np
import numpy.linalg as npl
import scipy.linalg as scl
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
import modem as mod
import util as utl

from version import versionstrg

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
blank = 1.e-30 # np.nan
rhoair = 1.e17

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


rhoair = 1.e+17

total = 0
ModDir_in = JACOPYAN_DATA + "/Peru/Misti/"
ModDir_out = ModDir_in + "/results_shuttle/"

ModFile_in = ModDir_in + "Misti10_best"    
ModFile_out = ModFile_in
ModFormat = "mod rlm" # "ubc"   
ModOrig = [-16.277300, -71.444397]# Misti


JacFile = ModDir_in +"Misti_best_Z5_nerr_sp-8"
JacFormat = "sparse"

ModModFormat = "mod rlm" # "ubc"

if not os.path.isdir(ModDir_out):
    print("File: %s does not exist, but will be created" % ModDir_out)
    os.mkdir(ModDir_out)



samples = 100
smoother = None  # ["uniform", 3] ["gaussian",0.5]

distribution = "regular"  # "random", "regular"

#            geo    act   dlog10rho     axes                  angles
basebody = ["box", "add", 0.2,       3000., 1000., 2000.,    0., 0., 0.]


utm_x, utm_y = utl.proj_latlon_to_utm(ModOrig[0], ModOrig[1], utm_zone=32631)
utmcenter = [utm_x, utm_y, 0.]

total = 0.
start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(ModFile_in, ".rho",trans="log10")
mdims = np.shape(rho)
aircells = np.where(rho>np.log10(rhoair/10.))
jacmask = jac.set_airmask(rho=rho, aircells=aircells, blank=np.log10(blank), flat = False, out=True)
jacflat = jacmask.flatten(order="F")
elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile_in + ".rho"))

start = time.perf_counter()
print("Reading Jacobian from "+JacFile)
if "sp" in JacFormat:
    Jac = scs.load_npz(JacFile +"_jac.npz")
    normalized = True
    tmp = np.load(JacFile +"_info.npz", allow_pickle=True)
    Freqs = tmp["Freq"]
    Comps = tmp["Comp"]
    Sites = tmp["Site"]
    Dtype = tmp["DTyp"]
    print(np.unique(Dtype))

else:  
    
    Jac, tmp = mod.read_jac(JacFile + ".jac")    
    normalized = False
    Data, Sites, Freqs, Comps, Dtype, Head = mod.read_data_jac(JacFile + "_jac.dat")
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)
     
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, JacFile))
total = total + elapsed
print("Full Jacobian")
jac.print_stats(jac=Jac, jacmask=jacflat)
print("\n")               
print("\n")



rho = mod.prepare_model(rho, rhoair=rhoair)

for ibody in range(samples):
    body = basebody.copy()
    
     


    rho_new = mod.insert_body(dx, dy, dz, rho, body) 
    rho_new[aircells] = rhoair
    
    ModFile = ModFile_out + "_" + body[0] + \
        str(ibody) + "_" + smoother[0] + ".rho"
        
    Header = "# "+ModFile
       
    if "mod" in ModFormat.lower():
        # for modem_readable files

        mod.write_mod(ModFile, modext="_rho_new.rho",
                      dx=dx, dy=dy, dz=dz, mval=rho_new,
                      reference=refmod, mvalair=blank, aircells=aircells, header=Header)
        print(" Cell rho_newumes (ModEM format) written to "+ModFile)
        
    if "ubc" in ModFormat.lower():
        elev = -refmod[2]
        refubc =  [ModOrig[0], ModOrig[1], elev]
        mod.write_ubc(ModFile, modext="_ubc.rho_new", mshext="_ubc.msh",
                      dx=dx, dy=dy, dz=dz, mval=rho_new, reference=refubc, mvalair=blank, aircells=aircells, header=Header)
        print(" Cell rho_newumes (UBC format) written to "+ModFile)
  
    if "rlm" in ModFormat.lower():
        mod.write_rlm(ModFile, modext="_rho_new.rlm", 
                      dx=dx, dy=dy, dz=dz, mval=rho_new, reference=refmod, mvalair=blank, aircells=aircells, comment=Header)
        print(" Cell rho_newumes (UBC format) written to "+ModFile)     


    elapsed = (time.perf_counter() - start)
    print(
        " Used %7.4f s for processing/writing model to %s " %
        (elapsed, ModFile))
    print("\n")


total = total + elapsed
print(" Total time used:  %f s " % (total))
