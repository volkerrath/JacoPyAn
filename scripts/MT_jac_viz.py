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

import matplotlib as mp
import matplotlib.colors as mco
import matplotlib.pyplot as mpl
import matplotlib.cm as cm


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
rhoair = 1.e17

InpFormat = "sparse"
OutFormat = "mod ubc" 

WorkDir = JACOPYAN_ROOT+"/work/"
WorkName = "UBI_ZPT_nerr_sp-8"
MFile   = WorkDir + "UBI_best.rho"
MPad=[14, 14 , 14, 14, 0, 71]

MOrig = [-16.345800, -70.908249]


JFile = WorkDir + "UBI_ZPT_nerr_sp-8"


Splits = ["comp", "site", "freq"]

FreqBands = [ [0.0001, 0.01], [0.01, 0.1], [0.1, 1.], [1., 100.], [100., 1000.], [1000., 10000.]]



Type = "euc"
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

Transform = [ "max", "sqr"]
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

start = time.perf_counter()
dx, dy, dz, rho, refmod, _, vcell = mod.read_mod(MFile, trans="linear", volumes=True)
V= vcell.flatten(order="F")
elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

dims = np.shape(rho)
sdims = np.size(rho)

aircells = np.where(rho>rhoair/10)


# TSTFile = WorkDir+WorkName+"0_MaskTest.rho"
# mod.write_mod(TSTFile, dx, dy, dz, rho, refmod, trans="LINEAR", mvalair=blank, aircells=aircells)


jacmask = jac.set_mask(rho=rho, pad=MPad, blank= blank, flat = False, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)

# rhotest = jacmask.reshape(dims)*rho
# TSTFile = WorkDir+WorkName+"1_MaskTest.rho"
# mod.write_mod(TSTFile, dx, dy, dz, rhotest, refmod, trans="LINEAR", mvalair=blank, aircells=aircells)


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
    Dtype = tmp["Dtype"]
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
jm = jacmask.flatten(order="F")
print(JFile+" minimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))
#mx = np.amax(np.abs(Jac*jm))
#mn = np.amin(np.abs(Jac*jm))
#print(JFile+" minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx))
# print(JFile+" number of elements in masked Jacobian is "+str(np.count_nonzero(~np.isfinite(Jac))))
# print( np.count_nonzero(~np.isnan(jacmask))*np.shape(Jac)[0])

start = time.perf_counter()
#print("Jac ", np.shape(Jac))
#Jac = Jac.toarray()
SensTmp = jac.calc_sensitivity(Jac,
                     Type = Type, OutInfo=False)
SensTot = jac.transform_sensitivity(S=SensTmp, V=V,
                          Transform=Transform, OutInfo=False)

SensFile = WorkDir+WorkName+"_"+Type+"_"+"_".join(Transform)
Head = (WorkName+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
# SensTot = SensTot.toarray()
# print(type(SensTot))
# print(np.shape(SensTot))   
S = SensTot.reshape(dims, order="F")
# S = np.reshape(SensTot, dims, order="F")

if "mod" in OutFormat.lower():
    mod.write_mod(SensFile, ModExt=".sns",
                  dx=dx, dy=dy, dz=dz, mval=S,
                  reference=refmod, mvalair=rhoair, aircells=aircells, header=Head)
    print(" Sensitivities (ModEM format) written to "+SensFile)
    
if "ubc" in OutFormat.lower():
    elev = -refmod[2]
    refubc =  [MOrig[0], MOrig[1], elev]
    mod.write_ubc(SensFile, ModExt=".sns",
                  dx=dx, dy=dy, dz=dz, mval=S, reference=refubc, mvalair=rhoair, aircells=aircells, header=Head)
    print(" Sensitivities (UBC format) written to "+SensFile)
    
    
V = V.reshape(dims, order="F")   
if "mod" in OutFormat.lower():
    mod.write_mod(SensFile, ModExt=".vol",
                  dx=dx, dy=dy, dz=dz, mval=V,
                  reference=refmod, mvalair=rhoair, aircells=aircells, header=Head)
    print(" Cell volumes (ModEM format) written to "+SensFile)
    
if "ubc" in OutFormat.lower():
    elev = -refmod[2]
    refubc =  [MOrig[0], MOrig[1], elev]
    mod.write_ubc(SensFile, ModExt=".vol",
                  dx=dx, dy=dy, dz=dz, mval=V, reference=refubc, mvalair=rhoair, aircells=aircells, header=Head)
    print(" Cell volumes (UBC format) written to "+SensFile)
  
  
elapsed = time.perf_counter() - start
print(" Used %7.4f s for full sensitivities " % (elapsed))
        
    #       for j in range(lower_bound_on_rows, upper_bound_on_rows): nums.append(j)
    # partial_matrix = my_matrix[nums, :] 

    # plt.matshow(partial_matrix, fignum=100)
    # plt.gca().set_aspect('auto')
    # plt.savefig('filename.png', dpi=600)
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
            JacTmp = Jac[np.where(Comps == icmp)]
            SensTmp = jac.calc_sensitivity(JacTmp,
                         Type = Type, OutInfo=False)
            SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                              Transform=Transform, OutInfo=False)
            SensFile = WorkDir+WorkName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)
            Head = (WorkName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
            S = np.reshape(SensTot, dims, order="F")
            if "mod" in OutFormat.lower():
                mod.write_mod(SensFile, ".sns",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refmod, mvalair=rhoair, aircells=aircells, header=Head)
                print(" Component sensitivities (ModEM format) written to "+SensFile)
                
            if "ubc" in OutFormat.lower():
                elev = -refmod[2]
                refubc =  [MOrig[0], MOrig[1], elev]
                mod.write_ubc(SensFile, ModExt=".sns",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refubc, mvalair=rhoair, aircells=aircells, header=Head)
                print(" Component sensitivities (UBC format) written to "+SensFile)
            
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for comp sensitivities " % (elapsed))        
        print("\n")
    
    if "site" in Split.lower():
        start = time.perf_counter()
        

        SiteNums = Sites[np.sort(np.unique(Sites, return_index=True)[1])] 
        SiteNames = Site[np.sort(np.unique(Site, return_index=True)[1])] 
    
        
        for isit in SiteNums:            
           JacTmp = Jac[np.where(Sites == isit)]
           SensTmp = jac.calc_sensitivity(JacTmp,
                        Type = Type, OutInfo=False)
           SensTmp = jac.transform_sensitivity(S=SensTmp, V=V,
                             Transform=Transform, OutInfo=False)
           SensFile = WorkDir+WorkName+"_"+SiteNames[isit-1].lower()+"_"+Type+"_"+"_".join(Transform)
           Head = (WorkName+"_"+SiteNames[isit-1].lower()+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
           S = np.reshape(SensTot, dims, order="F") 
           if "mod" in OutFormat.lower():
                mod.write_mod(SensFile, ModExt=".sns",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refmod, mvalair=rhoair, aircells=aircells, header=Head)
                print(" Site sensitivities (ModEM format) written to "+SensFile)
                
           if "ubc" in OutFormat.lower():
                elev = -refmod[2]
                refubc =  [MOrig[0], MOrig[1], elev]
                mod.write_ubc(SensFile, ModExt=".sns",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refubc, mvalair=rhoair, aircells=aircells, header=Head)
                print(" Site sensitivities (UBC format) written to "+SensFile)           
           
        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for site sensitivities " % (elapsed))
        print("\n")
    
    if "freq" in Split.lower():
        start = time.perf_counter()
        
        nF = len(FreqBands)
       
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
           SensFile = WorkDir+WorkName+"_freqband"+lowstr+"_to_"+uppstr+"_"+Type+"_"+"_".join(Transform)
           Head = (WorkName+"_freqband"+lowstr+"to"+uppstr+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
           S = np.reshape(SensTot, dims, order="F") 
           if "mod" in OutFormat.lower():
               mod.write_mod(SensFile, ModExt=".sns",
                             dx=dx, dy=dy, dz=dz, mval=S,
                             reference=refmod, mvalair=rhoair, aircells=aircells, header=Head)
               print(" Frequency band sensitivities (ModEM format) written to "+SensFile)
     
           if "ubc" in OutFormat.lower():
               elev = -refmod[2]
               refubc =  [MOrig[0], MOrig[1], elev]
               mod.write_ubc(SensFile, ModExt=".sns",
                             dx=dx, dy=dy, dz=dz, mval=S,
                             reference=refubc, mvalair=rhoair, aircells=aircells, header=Head)
               print(" Frequency band sensitivities (UBC format) written to "+SensFile)   
           

        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for Freq sensitivities " % (elapsed))
        print("\n")
