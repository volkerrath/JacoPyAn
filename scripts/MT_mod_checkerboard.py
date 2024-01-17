#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 06:18:17 2024

@author: vrath
"""
import os
import sys
import time

import numpy as np


JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import modem as mod

rhoair = np.log10(1.e17)
blank = np.log10(rhoair)

WorkDir = JACOPYAN_DATA+"/Annecy/Jacobians/"
MFile = WorkDir+"ANN_best"



bs = [3, 3, 3]
ba = [0.2, -0.2]
pad = [5, 5, 10]
mindepth = 3

idstr = "_logadd_"+str(ba[0])+"_"+str(ba[1])+"_size"+str(bs[0])+"_"+str(bs[1])+"_"+str(bs[2])

start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(MFile, trans="log10", volumes=True)
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

aircells = np.where(rho>np.log10(rhoair)-1)

xpad, ypad, zpad = 5, 5, 10
mindepth = 3

rho_new = mod.generate_checkerboard(rho=rho, 
                          pad=pad, mindepth=mindepth, 
                          bs =bs, ba=ba, out=True)
    
rho_new[aircells] = rhoair
rho_new = np.power(10.,rho_new)
print(rho_new)

header = "checkerboard size = "+str(bs)+" cells, added values are "+str(ba)
mod.write_mod(MFile, modext=idstr+".rho",
                dx=dx, dy=dy, dz=dz, mval=rho_new,trans="log10",
                reference=refmod, mvalair=rhoair, aircells=aircells, header=header)
