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


# Task = "merge"
MergedFile = "UBI_ZPT_sp-8"
MFiles = [WorkDir+"//UBI_Z_nerr_sp-8", WorkDir+"//UBI_P_nerr_sp-8",WorkDir+"//UBI_T_nerr_sp-8",]
nF = np.size(MFiles)
print(" The following files will be merged:")
print(MFiles)

Task = "split"
SFile = WorkDir+"UBI_ZPT_sp-8"
Split = ["comp", "site", "freq"]
print(SFile)
print(" The file will be split into components:")
print(Split)

FreqBands = [ [0.0001, 0.01], [0.01, 0.1], [0.1, 1.], [1., 100.], [100., 1000.], [1000., 10000.]]


if "mer" in Task.lower():
    
    
    for ifile in np.arange(nF):     
       
        start =time.time()
        print("\nReading Data from "+MFiles[ifile])
        Jac = scs.load_npz(MFiles[ifile] +"_jac.npz")
        
        
        tmp = np.load( MFiles[ifile] +"_info.npz", allow_pickle=True)
        Freq = tmp["Freq"]
        Comp = tmp["Comp"]
        Site = tmp["Site"]
        Dtyp = tmp["Dtyp"]
        Data = tmp["Data"]
        Scale = tmp["Scale"]
        Info = tmp["Info"]
        
        elapsed = time.time() - start
        print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, MFiles[ifile]))
        
        
       
        if ifile==0:
            Jac_merged = Jac 
            print(ifile, type(Jac_merged), type(Jac), np.shape(Jac_merged))
            Data_merged = Data 
            Site_merged = Site
            Freq_merged = Freq
            Comp_merged = Comp
            Dtyp_merged = Dtyp 
            Infblk = Info
            Scales = Scale
        else:
            Jac_merged = scs.vstack((Jac_merged, Jac))            
            print(ifile, type(Jac_merged), np.shape(Jac_merged))
            Data_merged = np.vstack((Data_merged, Data)) 
            Site_merged = np.hstack((Site_merged, Site))
            Freq_merged = np.hstack((Freq_merged, Freq))
            Comp_merged = np.hstack((Comp_merged, Comp))
            Dtyp_merged = np.hstack((Dtyp_merged, Dtyp))
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

    start =time.time()
    print("\nReading Data from "+SFile)
    Jac = scs.load_npz(SFile+"_jac.npz")
    if Jac.issparse(): sparse= True
    tmp = np.load(SFile+"_info.npz", allow_pickle=True)
    Freq = tmp["Freq"]
    Comp = tmp["Comp"]
    Site = tmp["Site"]
    Dtyp = tmp["Dtyp"]
    Data = tmp["Data"]
    Scal = tmp["Scale"]
    Info = tmp["Info"]
    
    if "fre" in Split.lower():
                   
        start = time.time()
        
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
           
           
           Name = SFile+"_freqband"+lowstr+"_to_"+uppstr
           Head =os.path.basename(Name).replace("_", " | ")              
           
     
           np.savez_compressed( Name +"_info.npz", Freq=Freq, Data=Data, Site=Site, Comp=Comp, 
                                Info=Info, Dtype=Dtype, Scale=Scale, allow_pickle=True)
           if scs.issparse(JacTmp):
                scs.save_npz( Name +"_jac.npz", matrix=JacTmp) #, compressed=True)
           else:
                np.savez_compressed(Name +"_jac.npz", JacTmp)
            
       
               
        elapsed = time.time() - start
        print(" Used %7.4f s for splitting into frequency bands " % (elapsed))        
        print("\n")
        

    if "com" in Split.lower():
           
        start = time.time()
    
        """
        Full_Impedance              = 1
        Off_Diagonal_Impedance      = 2
        Full_Vertical_Components    = 3
        Full_Interstation_TF        = 4
        Off_Diagonal_Rho_Phase      = 5
        Phase_Tensor                = 6
        """
        compstr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]
    
        ExistType = np.unique(Dtyp)
        
        for icmp in ExistType:
            
            print(icmp)
            JacTmp = Jac[np.where(Comp == icmp)]

            Name = SFile+"_dtype"+compstr[icmp-1]
            Head =os.path.basename(Name).replace("_", " | ")
    
            np.savez_compressed( Name +"_info.npz", Freq=Freq, Data=Data, Site=Site, Comp=Comp, 
                                Info=Info, Dtype=Dtyp, Scale=Scale, allow_pickle=True)
            if scs.issparse(JacTmp):
                scs.save_npz( Name +"_jac.npz", matrix=JacTmp) #, compressed=True)
            else:
                np.savez_compressed(Name +"_jac.npz", JacTmp)
  
                
        elapsed = time.time() - start
        print(" Used %7.4f s for splitting into components " % (elapsed))        
        print("\n")
        
    if "sit" in Split.lower():          
        pass
        # start = time.time()
        
        # SiteNums = Sites[np.sort(np.unique(Sites, return_index=True)[1])] 
        # SiteNames = Site[np.sort(np.unique(Site, return_index=True)[1])] 
    
        
        # for isit in SiteNums:        
            
        #     JacTmp = Jac[np.where(Sites == isit)]
 
        #     SensFile = WorkDir+WorkName+"_"+SiteNames[isit-1].lower()+"_"+Type+"_"+"_".join(Transform)
        #     Head = (WorkName+"_"+SiteNames[isit-1].lower()+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")   

            
        #     Name = SFile+"_dtype"+compstr[icmp-1]
        #     Head = Name.replace("_", " | ")
            
        #     np.savez_compressed( Name +"_info.npz", Freq=Freq, Data=Data, Site=Site, Comp=Comp, 
        #                         Info=Info, Dtype=Dtype, Scale=Scale, allow_pickle=True)
        #     if scs.issparse(JacTmp):
        #         scs.save_npz( Name +"_jac.npz", matrix=JacTmp) #, compressed=True)
        #     else:
        #         np.savez_compressed(Name +"_jac.npz", JacTmp)
  
                
        # elapsed = time.time() - start
        # print(" Used %7.4f s for splitting into sites " % (elapsed))        
        # print("\n")
        
