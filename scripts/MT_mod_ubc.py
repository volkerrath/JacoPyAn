#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:17:16 2023

@author: vrath
"""
import os
import sys
import numpy as np

from sys import exit as error
# import struct
import time
from datetime import datetime
import warnings


JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]
JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]

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

debug = True

rhoair = 1.e17
blank = rhoair


Task = "Mod2UBC"

if "mod2ubc" in Task.lower():
    # MOD_Data  =  "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014.dat"
    MOD_Modl  =  "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014"
    MPad=[0, 0, 0, 0, 0, 0]
    UBC_Modl = MOD_Modl
    lat, lon =  -16.346,  -70.908

    start = time.time()
    dx, dy, dz, rho, refmod, _, vcell = mod.read_model_mod(MOD_Modl, ".rho",trans="linear", volumes=True)
    dims = np.shape(rho)
    sdims = np.size(rho)
    
    elapsed = time.time() - start
    print(" Used %7.4f s for reading MOD model from %s " % (elapsed, MOD_Modl))

   
    aircells = np.where(rho>rhoair/10)

    if debug: 
        ModExt = ".rho_debug"
        mod.write_model_mod(MOD_Modl, ModExt,
                            dx, dy, dz, rho, 
                            reference=refmod, trans="LINEAR", mvalair=blank, aircells=aircells)

    start = time.time()    
    elev = -refmod[2]
    refcenter =  [lat, lon, elev]
    MshExt = ".mesh"
    ModExt = ".mod"
    mod.write_model_ubc(UBC_Modl, MshExt, ModExt, 
                        dx, dy, dz, rho, refcenter, mvalair=blank, aircells=aircells)
    elapsed = time.time() - start
    print(" Used %7.4f s for Writing UBC model to %s " % (elapsed, UBC_Modl))
    

if "ubc2mod" in Task.lower():
    UBC_Modl  =  "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014"
    MOD_Modl  =  UBC_Modl
    MshExt = ".mesh"
    ModExt = ".mod"
    
    start = time.time()
    dx, dy, dz, val, refubc, _, vcell = mod.read_model_ubc(UBC_Modl, MshExt, ModExt,  
                                                        volumes=True, out=True)
    elapsed = time.time() - start
    print(" Used %7.4f s for reading UBC model from %s " % (elapsed, UBC_Modl))


    ModExt = ".rho"
    mod.write_model_mod(MOD_Modl, ModExt,
                        dx, dy, dz, rho, 
                        reference=refmod, trans="LINEAR", mvalair=blank, aircells=aircells)


