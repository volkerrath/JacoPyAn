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
Blank = 1.e-30 # np.nan
Rhoair = 1.e17

InpFormat = "sparse"
OutFormat = "mod" # "ubc"
# for 3d-Grid:
ModExt = "_sns.rho"

# WorkDir = JACOPYAN_DATA+"/Annecy/Jacobians/"
# WorkDir = JACOPYAN_DATA+"/Peru/Sabancaya/"
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"

if not WorkDir.endswith("/"):
    WorkDir = WorkDir+"/"
    
# MFile = WorkDir + "SABA8_best.rho"
MFile = WorkDir + "UBI9_best"
# necessary, but not relevant  for synthetic model 
MOrig = [-16.345800 -70.908249]

JacName = "UBI9_ZPTss_nerr_sp-8"
JFile = WorkDir + JacName

VolExtract = False
if VolExtract:
    VolFile = MFile
    VolFmt = ""
    
TopoExtract = True
if TopoExtract:
    TopoFile = WorkDir + "UBI9_Topo.dat"
    TopoFmt = ""


Splits = ["comp", "site", "freq"]

PerIntervals = [ 
                [0.0001, 0.001], 
                [0.001, 0.01], 
                [0.01, 0.1], 
                [0.1, 1.], 
                [1., 10.], 
                [10., 100.], 
                [100., 1000.], 
                [1000., 10000.]
                ]



# Type = "raw"
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


# Transform = [ "max"]
Transform = [ "sqr","max"]

"""
Transform sensitivities. 
Options:
    Transform = "vol"           Normalize by the values optional array vol ("volume"), 
                                i.e in our case layer thickness. This should always 
                                be the first value in Transform list.
    Transform = "max"           Normalize by maximum (absolute) value.
    Transform = "sur"           Normalize by surface value.
    Transform = "sqr"           Take the square root. Only usefull for euc sensitivities. 
    Transform = "log"           Take the logaritm. This should always be the 
                                last value in Transform list
"""

SensDir = WorkDir+"/sens_"+Type.lower()+"/"
if not SensDir.endswith("/"):
    SensDir = SensDir+"/"
if not os.path.isdir(SensDir):
    print("File: %s does not exist, but will be created" % SensDir)
    os.mkdir(SensDir)


total = 0.0

start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(MFile, trans="linear", volumes=True)
elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

aircells = np.where(rho>Rhoair/10)



if TopoExtract:
    
    xcnt, ycnt, topo = mod.get_topo(dx=dx, dy=dy, dz=dz, mval=rho, ref= refmod,
             mvalair = 1.e17, out=True)
    
    if os.path.isfile(TopoFile):
        os.remove(TopoFile)
        
    with open(TopoFile, "a") as f:

        for ii in np.arange(len(dx)):
            for jj in np.arange(len(dy)):
                line = str(xcnt[ii])+", "+str(ycnt[jj])+", "+str(topo[ii,jj])+"\n"
                f.write(line)

        
    
if VolExtract or ("size" in Transform):
    vol = mod.get_volumes(dx=dx, dy=dy, dz=dz, mval=rho, out=True)
    print(np.shape(vol))
    Header = "# "+MFile
    
    if "mod" in OutFormat.lower():
        # for modem_readable files

        mod.write_mod(VolFile, modext="_vol.rho",
                      dx=dx, dy=dy, dz=dz, mval=vol,
                      reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
        print(" Cell volumes (ModEM format) written to "+VolFile)
        
    if "ubc" in OutFormat.lower():
        elev = -refmod[2]
        refubc =  [MOrig[0], MOrig[1], elev]
        mod.write_ubc(VolFile, modext="_ubc.vol", mshext="_ubc.msh",
                      dx=dx, dy=dy, dz=dz, mval=vol, reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
        print(" Cell volumes (UBC format) written to "+VolFile)
  
    
else:
    vol = np.array([])
    
        
mdims = np.shape(rho)

aircells = np.where(rho>Rhoair/10)
jacmask = jac.set_airmask(rho=rho, aircells=aircells, blank=Blank, flat = False, out=True)
jacflat = jacmask.flatten(order="F")
name, ext = os.path.splitext(MFile)

# ofile = name
# Header = JacName
# trans = "LINEAR"

# mod.write_mod(ofile, modext="_mod.rho", trans = trans,
#                   dx=dx, dy=dy, dz=dz, mval=rho,
#                   reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
# print(" Model (ModEM format) written to "+ofile)
    
# # elev = -refmod[2]
# # refubc =  [MOrig[0], MOrig[1], elev]
# # mod.write_ubc(OFile, modext="_rho_ubc.mod", mshext="_rho_ubc.msh",
# #                   dx=dx, dy=dy, dz=dz, mval=rho, reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
# # print(" Model (UBC format) written to "+ofile)
    
# TSTFile = WorkDir+JacName+"0_MaskTest"
# mod.write_mod(TSTFile, modext="_mod.rho", trans = trans,
#             dx=dx, dy=dy, dz=dz, mval=rho, reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
# rhotest = jacmask.reshape(dims)*rho
# TSTFile = WorkDir+JacName+"1_MaskTest"
# mod.write_mod(TSTFile, modext="_mod.rho", trans = trans,
#             dx=dx, dy=dy, dz=dz, mval=rhotest, reference=refmod, mvalair=Blank, aircells=aircells, header=Header)


# name, ext = os.path.splitext(JFile)

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

print("Full Jacobian")
jac.print_stats(jac=Jac, jacmask=jacflat)
print("\n")               
print("\n")
        

start = time.perf_counter()


SensTmp = jac.calc_sensitivity(Jac,
                     Type = Type, OutInfo=False)
SensTot = jac.transform_sensitivity(S=SensTmp, vol=vol, 
                          Transform=Transform, OutInfo=False)

SensFile = SensDir+JacName+"_total_"+Type+"_"+"_".join(Transform)
Header = ("# "+JacName+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
S = SensTot.reshape(mdims, order="F")

if "mod" in OutFormat.lower():
    mod.write_mod(SensFile, modext=ModExt,
                  dx=dx, dy=dy, dz=dz, mval=S,
                  reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
    print(" Sensitivities (ModEM format) written to "+SensFile)
    
if "ubc" in OutFormat.lower():
    elev = -refmod[2]
    refubc =  [MOrig[0], MOrig[1], elev]
    mod.write_ubc(SensFile, modext="_ubc.sns", mshext="_ubc.msh",
                  dx=dx, dy=dy, dz=dz, mval=S, reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
    print(" Sensitivities (UBC format) written to "+SensFile)
    
  
  
elapsed = time.perf_counter() - start
print(" Used %7.4f s for full sensitivities " % (elapsed))
        
        
for Split in Splits:
        
    if "comp" in Split.lower():
        
        start = time.perf_counter()
    
        """
        Full_Impedance              = 1
        Off_Diagonal_Impedance      = 2
        Full_volertical_Components    = 3
        Full_Interstation_TF        = 4
        Off_Diagonal_Rho_Phase      = 5
        Phase_Tensor                = 6
        """
        compstr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]
    
        ExistType = np.unique(Dtype)
        
        for icmp in ExistType:
            indices = np.where(Dtype == icmp)
            JacTmp = Jac[indices]
            print("Component: ",icmp)
            jac.print_stats(jac=JacTmp, jacmask=jacflat)
            print("\n")
            
            SensTmp = jac.calc_sensitivity(JacTmp,
                         Type = Type, OutInfo=False)
            SensTmp = jac.transform_sensitivity(S=SensTmp, vol=vol,
                              Transform=Transform, OutInfo=False)
            SensFile = SensDir+JacName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)
            Header = ("# "+JacName+"_"+compstr[icmp-1]+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
            S = np.reshape(SensTmp, mdims, order="F")
            if "mod" in OutFormat.lower():
                mod.write_mod(SensFile, ModExt,
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
                print(" Component sensitivities (ModEM format) written to "+SensFile)
                
            if "ubc" in OutFormat.lower():
                elev = -refmod[2]
                refubc =  [MOrig[0], MOrig[1], elev]
                mod.write_ubc(SensFile, modext="_ubc.sns" ,mshext="_ubc.msh",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
                print(" Component sensitivities (UBC format) written to "+SensFile)
            
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for comp sensitivities " % (elapsed))        
        print("\n")
    
    if "site" in Split.lower():
        start = time.perf_counter()
        

        SiteNames = Sites[np.sort(np.unique(Sites, return_index=True)[1])] 
    
        
        for sit in SiteNames:        
           indices = np.where(sit==Sites)
           JacTmp = Jac[indices]
           print("Site: ",sit)
           jac.print_stats(jac=JacTmp, jacmask=jacflat)
           print("\n")
           
           SensTmp = jac.calc_sensitivity(JacTmp,
                        Type = Type, OutInfo=False)
           SensTmp = jac.transform_sensitivity(S=SensTmp, vol=vol,
                             Transform=Transform, OutInfo=False)
           SensFile = SensDir+JacName+"_"+sit.lower()+"_"+Type+"_"+"_".join(Transform)
           Header = ("# "+JacName+"_"+sit.lower()+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
           S = np.reshape(SensTmp, mdims, order="F") 
           if "mod" in OutFormat.lower():
                mod.write_mod(SensFile, modext=ModExt,
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
                print(" Site sensitivities (ModEM format) written to "+SensFile)
                
           if "ubc" in OutFormat.lower():
                elev = -refmod[2]
                refubc =  [MOrig[0], MOrig[1], elev]
                mod.write_ubc(SensFile, modext="_ubc.sns", mshext="_ubc.msh",
                              dx=dx, dy=dy, dz=dz, mval=S,
                              reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
                print(" Site sensitivities (UBC format) written to "+SensFile)           
           
        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for site sensitivities " % (elapsed))
        print("\n")
    
    if "freq" in Split.lower():
        start = time.perf_counter()
        
        nF = len(PerIntervals)
        
        for ibnd in np.arange(nF):  
         
           lowstr=str(1./PerIntervals[ibnd][0])+"Hz"            
           uppstr=str(1./PerIntervals[ibnd][1])+"Hz"                   

           
           indices = np.where((Freqs>=PerIntervals[ibnd][0]) & (Freqs<PerIntervals[ibnd][1]))

        
           JacTmp = Jac[indices]
           if np.shape(JacTmp)[0] > 0:
               print("Freqband: ", lowstr, "to", uppstr)
               jac.print_stats(jac=JacTmp, jacmask=jacflat)
               print("\n")

               SensTmp = jac.calc_sensitivity(JacTmp,
                            Type = Type, OutInfo=False)
               SensTmp = jac.transform_sensitivity(S=SensTmp, vol=vol,
                                 Transform=Transform, OutInfo=False)
               SensFile = SensDir+JacName+"_freqband"+lowstr+"_to_"+uppstr+"_"+Type+"_"+"_".join(Transform)
               Header = ("# "+JacName+"_freqband"+lowstr+"to"+uppstr+"_"+Type+"_"+"_".join(Transform)).replace("_", " | ")
               S = np.reshape(SensTmp, mdims, order="F") 
               if "mod" in OutFormat.lower():
                   mod.write_mod(SensFile, modext=ModExt,
                                 dx=dx, dy=dy, dz=dz, mval=S,
                                 reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
                   print(" Frequency band sensitivities (ModEM format) written to "+SensFile)
         
               if "ubc" in OutFormat.lower():
                   elev = -refmod[2]
                   refubc =  [MOrig[0], MOrig[1], elev]
                   mod.write_ubc(SensFile, modext="_ubc.sns", mshext="_ubc.msh",
                                 dx=dx, dy=dy, dz=dz, mval=S,
                                 reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
                   print(" Frequency band sensitivities (UBC format) written to "+SensFile)   
           else: 
                print("Frequency band is empty! Continue.")

        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for Freq sensitivities " % (elapsed))
        print("\n")
