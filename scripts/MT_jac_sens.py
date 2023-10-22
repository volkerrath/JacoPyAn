#!/usr/bin/env python3

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
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


gc.enable()

rng = np.random.default_rng()
blank = 1.e-30 # np.nan
rhoair = 1.e17

InpFormat = "sparse"
OutFormat = "mod ubc" 

WorkDir = JACOPYAN_ROOT+"/work/Sabancaya/"


MFile   = WorkDir + "SABA8_best.rho"
MPad=[10, 10 , 10, 10, 0, 20]
# MPad=[0, 0, 0, 0, 0, 0]

# 
MOrig = [-15.767401, -71.854095]

WorkName = "SABA8_Z_sp-8" #
# WorkName = "SABA8_T_sp-8"
# WorkName = "SABA8_P_sp-8"

JFile = WorkDir + WorkName


Splits = ["comp", "site", "freq"]

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


# Transform = [ "max", "sqr"]
Transform = [ "max"]
"""
Transform sensitivities. 
Options:
    Transform = "siz",          Normalize by the values optional array V ("volume"), 
                                i.e in our case layer thickness. This should always 
                                be the first value in Transform list.
    Transform = "max"           Normalize by maximum (absolute) value.
    Transform = "sur"           Normalize by surface value.
    Transform = "sqr"           Take the square root. Only usefull for euc sensitivities. 
    Transform = "log"           Take the logaritm. This should always be the 
                                last value in Transform list
"""

total = 0.0

start = time.perf_counter()
dx, dy, dz, rho, refmod, _, vcell = mod.read_mod(MFile, trans="linear", volumes=True)
V= vcell.flatten(order="F")
elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

dims = np.shape(rho)
sdims = np.size(rho)

aircells = np.where(rho>rhoair/10)

name, ext = os.path.splitext(MFile)
OFile = name
Head = WorkName

mod.write_mod(OFile, ModExt="_mod.rho", trans = "LOGE",
                  dx=dx, dy=dy, dz=dz, mval=rho,
                  reference=refmod, mvalair=blank, aircells=aircells, header=Head)
print(" Model (ModEM format) written to "+OFile)
    
elev = -refmod[2]
refubc =  [MOrig[0], MOrig[1], elev]
mod.write_ubc(OFile, ModExt="_rho_ubc.mod", MshExt="_rho_ubc.msh",
                  dx=dx, dy=dy, dz=dz, mval=rho, reference=refubc, mvalair=blank, aircells=aircells, header=Head)
print(" Model (UBC format) written to "+OFile)
    
TSTFile = WorkDir+WorkName+"0_MaskTest.rho"
mod.write_mod(TSTFile, dx, dy, dz, rho, refmod, trans="LOGE", mvalair=blank, aircells=aircells)


jacmask = jac.set_mask(rho=rho, pad=MPad, blank= blank, flat = False, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)

rhotest = jacmask.reshape(dims)*rho
TSTFile = WorkDir+WorkName+"1_MaskTest.rho"
mod.write_mod(TSTFile, dx, dy, dz, rhotest, refmod, trans="LOGE", mvalair=blank, aircells=aircells)


name, ext = os.path.splitext(JFile)

start = time.perf_counter()
print("Reading Jacobian from "+JFile)

if "spa" in InpFormat:
    Jac = scs.load_npz(JFile +"_jac.npz")
    normalized = True
    
    tmp = np.load( JFile +"_info.npz", allow_pickle=True)
    Freqs = tmp["Freq"]
    Comps = tmp["Comp"]
    Sites = tmp["Site"]
    Dtype = tmp["DTyp"]
    print(np.unique(Dtype))

else:  
    
    Jac, tmp = mod.read_jac(JFile + ".jac")    
    normalized = False
    Freqs = tmp[:,0]
    Comps = tmp[:,1]
    Sites = tmp[:,2]
    Data, Site, Freq, Comp, Dtype, Head = mod.read_data_jac(JFile + "_jac.dat")
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)
    
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, JFile))
total = total + elapsed

mx = np.amax(np.abs(Jac))
mn = np.amin(np.abs(Jac))
print(JFile+" minimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))
jm = jacmask.flatten(order="F")
mx = np.amax(np.abs(Jac*jm))
mn = np.amin(np.abs(Jac*jm))
print(JFile+" minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx))
# print(JFile+" number of elements in masked Jacobian is "+str(np.count_nonzero(~np.isfinite(Jac))))
# print( np.count_nonzero(~np.isnan(jacmask))*np.shape(Jac)[0])

start = time.perf_counter()

SensTmp = jac.calc_sensitivity(Jac,
                     Type = Type, OutInfo=False)
SensTot = jac.transform_sensitivity(S=SensTmp, V=V,
                          Transform=Transform, OutInfo=False)

SensFile = WorkDir+WorkName+"_total_"+Type+"_"+"_".join(Transform)
Head = (WorkName+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
S = SensTot.reshape(dims, order="F")

if "mod" in OutFormat.lower():
    mod.write_mod(SensFile, ModExt="_mod.sns",
                  dx=dx, dy=dy, dz=dz, mval=S,
                  reference=refmod, mvalair=blank, aircells=aircells, header=Head)
    print(" Sensitivities (ModEM format) written to "+SensFile)
    
if "ubc" in OutFormat.lower():
    elev = -refmod[2]
    refubc =  [MOrig[0], MOrig[1], elev]
    mod.write_ubc(SensFile, ModExt="_ubc.sns", MshExt="_ubc.msh",
                  dx=dx, dy=dy, dz=dz, mval=S, reference=refubc, mvalair=blank, aircells=aircells, header=Head)
    print(" Sensitivities (UBC format) written to "+SensFile)
    
    
V = V.reshape(dims, order="F")   
if "mod" in OutFormat.lower():
    mod.write_mod(SensFile, ModExt="_mod.vol",
                  dx=dx, dy=dy, dz=dz, mval=V,
                  reference=refmod, mvalair=blank, aircells=aircells, header=Head)
    print(" Cell volumes (ModEM format) written to "+SensFile)
    
if "ubc" in OutFormat.lower():
    elev = -refmod[2]
    refubc =  [MOrig[0], MOrig[1], elev]
    mod.write_ubc(SensFile, ModExt="_ubc.vol", MshExt="_ubc.msh",
                  dx=dx, dy=dy, dz=dz, mval=V, reference=refubc, mvalair=blank, aircells=aircells, header=Head)
    print(" Cell volumes (UBC format) written to "+SensFile)
  
  
elapsed = time.perf_counter() - start
print(" Used %7.4f s for full sensitivities " % (elapsed))
        
        
for Split in Splits:
        
    if "comp" in Split.lower():
        
        start = time.perf_counter()
    
        """
        Full_Impedance              = 1
        Off_Diagonal_Impedance      = 2
        Full_Vertical_Components    = 3
        Full_Interstation_TF        = 4
        Off_Diagonal_Rho_Phase      = 5
        Phase_Tensor                = 6
        """
        compstr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]
    
        ExistType = np.unique(Dtype)
        
        for icmp in ExistType:
            print(icmp)
            JacTmp = Jac[np.where(Dtype == icmp)]
            SensTmp = jac.calc_sensitivity(JacTmp,
                         Type = Type, OutInfo=False)
            SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                              Transform=Transform, OutInfo=False)
            SensFile = WorkDir+WorkName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)
            Head = (WorkName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
            S = np.reshape(SensTmp, dims, order="F")
            if "mod" in OutFormat.lower():
                mod.write_mod(SensFile, "_mod.sns",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refmod, mvalair=blank, aircells=aircells, header=Head)
                print(" Component sensitivities (ModEM format) written to "+SensFile)
                
            if "ubc" in OutFormat.lower():
                elev = -refmod[2]
                refubc =  [MOrig[0], MOrig[1], elev]
                mod.write_ubc(SensFile, ModExt="_ubc.sns" ,MshExt="_ubc.msh",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refubc, mvalair=blank, aircells=aircells, header=Head)
                print(" Component sensitivities (UBC format) written to "+SensFile)
            
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for comp sensitivities " % (elapsed))        
        print("\n")
    
    if "site" in Split.lower():
        start = time.perf_counter()
        

        SiteNames = Sites[np.sort(np.unique(Sites, return_index=True)[1])] 
    
        
        for sit in SiteNames:            
           JacTmp = Jac[np.where(sit in Sites)]
           SensTmp = jac.calc_sensitivity(JacTmp,
                        Type = Type, OutInfo=False)
           SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                             Transform=Transform, OutInfo=False)
           SensFile = WorkDir+WorkName+"_"+sit.lower()+"_"+Type+"_"+"_".join(Transform)
           Head = (WorkName+"_"+sit.lower()+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
           S = np.reshape(SensTmp, dims, order="F") 
           if "mod" in OutFormat.lower():
                mod.write_mod(SensFile, ModExt="_mod.sns",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refmod, mvalair=blank, aircells=aircells, header=Head)
                print(" Site sensitivities (ModEM format) written to "+SensFile)
                
           if "ubc" in OutFormat.lower():
                elev = -refmod[2]
                refubc =  [MOrig[0], MOrig[1], elev]
                mod.write_ubc(SensFile, ModExt="_ubc.sns", MshExt="_ubc.msh",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refubc, mvalair=blank, aircells=aircells, header=Head)
                print(" Site sensitivities (UBC format) written to "+SensFile)           
           
        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for site sensitivities " % (elapsed))
        print("\n")
    
    if "freq" in Split.lower():
        start = time.perf_counter()
        
        nF = len(FreqBands)
        
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
           FreqList = np.where((Freqs>=FreqBands[ibnd][0]) & (Freqs<FreqBands[ibnd][1]))

        
           JacTmp = Jac[FreqList]
           SensTmp = jac.calc_sensitivity(JacTmp,
                        Type = Type, OutInfo=False)
           SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                             Transform=Transform, OutInfo=False)
           SensFile = WorkDir+WorkName+"_freqband"+lowstr+"_to_"+uppstr+"_"+Type+"_"+"_".join(Transform)
           Head = (WorkName+"_freqband"+lowstr+"to"+uppstr+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
           S = np.reshape(SensTmp, dims, order="F") 
           if "mod" in OutFormat.lower():
               mod.write_mod(SensFile, ModExt="_mod.sns",
                             dx=dx, dy=dy, dz=dz, mval=S,
                             reference=refmod, mvalair=blank, aircells=aircells, header=Head)
               print(" Frequency band sensitivities (ModEM format) written to "+SensFile)
     
           if "ubc" in OutFormat.lower():
               elev = -refmod[2]
               refubc =  [MOrig[0], MOrig[1], elev]
               mod.write_ubc(SensFile, ModExt="_ubc.sns", MshExt="_ubc.msh",
                             dx=dx, dy=dy, dz=dz, mval=S,
                             reference=refubc, mvalair=blank, aircells=aircells, header=Head)
               print(" Frequency band sensitivities (UBC format) written to "+SensFile)   
           

        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for Freq sensitivities " % (elapsed))
        print("\n")
