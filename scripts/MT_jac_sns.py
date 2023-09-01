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
from sys import exit as error
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
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")




gc.enable()

rng = np.random.default_rng()
blank = np.nan


outform = "LINEAR"
outform = outform.upper()

# UBINAS
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/UbiJac/"
WorkName = "UBI_best"
MFile   = WorkDir + "UBI_best.rho"
MPad=[14, 14 , 14, 14, 0, 71]

JFile = WorkDir + "UBI_best.jac"
DFile = WorkDir + "UBI_best_jac.dat"


Splits = [] #["comp", "site", "freq"]

FreqBands = [ [0.0001, 0.01], [0.01, 0.1], [0.1, 1.], [1., 100.], [100., 1000.], [1000., 10000.]]



Type = "raw"
"""
Calculate sensitivities.
Expects that Jacobian is already error-scaled, i.e Jac = C^(-1/2)*J.
Options:
    Type = "raw"     sensitivities summed along the data axis
    Type = "abs"     absolute sensitivities summed along the data axis
                     (often called coverage)
    Type = "euc"     squared sensitivities summed along the data axis.

Usesigma:
    if true, sensitivities with respect to sigma  are calculated.
"""

Transform = [ "siz", "max"]
"""
Transform sensitivities. 
Options:
    Transform = "siz",          Normalize by the values optional array V ("volume"), 
                                i.e in our case layer thickness. This should always 
                                be the first value in Transform list.
    Transform = "max"           Normalize by maximum value.
    Transform = "sur"           Normalize by surface value.
    Transform = "sqr"           Take the square root. Only usefull for euc sensitivities. 
    Transform = "log"           Take the logaritm. This should always be the 
                                last value in Transform list
"""

total = 0.0

start = time.time()
dx, dy, dz, rho, reference, _, vcell = mod.read_model_mod(MFile, trans="linear", volumes=True)
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


name, ext = os.path.splitext(JFile)
start =time.time()
print("\nReading Data from "+DFile)
Data, Site, Freq, Comp, Head = mod.read_data_jac(DFile)
elapsed = time.time() - start
print(" Used %7.4f s for reading Data from %s " % (elapsed, DFile))
total = total + elapsed

start = time.time()
print("Reading Jacobian from "+JFile)
Jac, Info = mod.read_jac(JFile)
elapsed = time.time() - start
print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile))
total = total + elapsed


start = time.time()
dsh = np.shape(Data)
err = np.reshape(Data[:, 5], (dsh[0], 1))
print(np.amin(err), np.amax(err))
Jac = jac.normalize_jac(Jac, err)
elapsed = time.time() - start
print(" Used %7.4f s for normalizing Jacobian with data error from %s " % (elapsed, DFile))

mx = np.nanmax(np.abs(Jac))
mn = np.nanmin(np.abs(Jac))
jm = jacmask.flatten(order="F")
print(JFile+" minimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))
mx = np.nanmax(np.abs(Jac*jm))
mn = np.nanmin(np.abs(Jac*jm))
print(JFile+" minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx))
# print(JFile+" number of elements in masked Jacobian is "+str(np.count_nonzero(~np.isfinite(Jac))))
# print( np.count_nonzero(~np.isnan(jacmask))*np.shape(Jac)[0])
V=vcell.flatten(order="F")
start = time.time()
print("Jac ", np.shape(Jac))
SensTmp = jac.calc_sensitivity(Jac,
                     Type = Type, OutInfo=False)
print("Sens ",np.shape(SensTmp))
SensTot = jac.transform_sensitivity(S=SensTmp, V=V,
                          Transform=Transform, OutInfo=False)

SensFile = WorkDir+WorkName+"_"+Type+"_"+"_".join(Transform)+".sns"
Head = (WorkName+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
S = np.reshape(SensTot, dims, order="F")
mod.write_model_mod(SensFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells, header=Head)
print(" Sensitivities written to "+SensFile)
elapsed = time.time() - start
print(" Used %7.4f s for sensitivities " % (elapsed))
        
        
for Split in Splits:
        
    if "comp" in Split.lower():
        
        start = time.time()
    
        """
        Full_Impedance              = 1
        Off_Diagonal_Impedance      = 2
        Full_Vertical_Components    = 3
        Full_Interstation_TF        = 4
        Off_Diagonal_Rho_Phase      = 5
        Phase_Tensor                = 6
        """
        compstr = ["zfull", "zoff", "tp", "rpoff", "pt"]
    
        Comps = Info[:,1]
        ExistComp = np.unique(Comps)
        
        for icmp in ExistComp:
            JacTmp = Jac[np.where(Comps == icmp)]
            SensTmp = jac.calc_sensitivity(JacTmp,
                         Type = Type, OutInfo=False)
            SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                              Transform=Transform, OutInfo=False)
            SensFile = WorkDir+WorkName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)+".sns"
            Head = (WorkName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
            S = np.reshape(SensTot, dims, order="F")
            mod.write_model_mod(SensFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells, header=Head)
            print(" Comp sensitivities written to "+SensFile)
            
        elapsed = time.time() - start
        print(" Used %7.4f s for comp sensitivities " % (elapsed))        
        print("\n")
    
    if "site" in Split.lower():
        start = time.time()
        
        Sites = Info[:,2]
        SiteNums = Sites[np.sort(np.unique(Sites, return_index=True)[1])] 
        SiteNames = Site[np.sort(np.unique(Site, return_index=True)[1])] 
    
        
        for isit in SiteNums:            
           JacTmp = Jac[np.where(Sites == isit)]
           SensTmp = jac.calc_sensitivity(JacTmp,
                        Type = Type, OutInfo=False)
           SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                             Transform=Transform, OutInfo=False)
           SensFile = WorkDir+WorkName+"_"+SiteNames[isit-1].lower()+"_"+Type+"_"+"_".join(Transform)+".sns"
           Head = (WorkName+"_"+SiteNames[isit-1].lower()+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
           S = np.reshape(SensTot, dims, order="F") 
           mod.write_model_mod(SensFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells, header=Head)
           print(" Site sensitivities written to "+SensFile)
        
        elapsed = time.time() - start
        print(" Used %7.4f s for site sensitivities " % (elapsed))
        print("\n")
    
    if "freq" in Split.lower():
        start = time.time()
        
        nF = len(FreqBands)
        Freqs = Info[:,0]
        FreqNums = Freqs[np.sort(np.unique(Freqs, return_index=True)[1])] 
        FreqValues = Freq[np.sort(np.unique(Freq, return_index=True)[1])] 
        
        for ibnd in np.arange(nF):  
           if np.log10(FreqBands[ibnd][0])<0.:
               lowstr=str(1./FreqBands[ibnd][0])+"s"
           else:
               lowstr=str(FreqBands[ibnd][0])+"Hz"
               
           if np.log10(FreqBands[ibnd][1])<0.:
               uppstr=str(1./FreqBands[ibnd][1])+"s"
           else:
               uppstr=str(FreqBands[ibnd][1])+"Hz"              
               

           freqstr = "" 
           FreqList = FreqNums[
               np.where((FreqValues>=FreqBands[ibnd][0]) & (FreqValues<FreqBands[ibnd][1]))
               ]
           print(FreqList)
        
           JacTmp = Jac[np.where(np.isin(Freqs, FreqList))]
           SensTmp = jac.calc_sensitivity(JacTmp,
                        Type = Type, OutInfo=False)
           SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                             Transform=Transform, OutInfo=False)
           SensFile = WorkDir+WorkName+"_freqband"+lowstr+"_to_"+uppstr+"_"+Type+"_"+"_".join(Transform)+".sns"
           Head = (WorkName+"_freqband"+lowstr+"to"+uppstr+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
           S = np.reshape(SensTot, dims, order="F") 
           mod.write_model_mod(SensFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells, header=Head)
           print(" Freq sensitivities written to "+SensFile)
        
        elapsed = time.time() - start
        print(" Used %7.4f s for Freq sensitivities " % (elapsed))
        print("\n")
