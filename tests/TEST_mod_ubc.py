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


DFile  =  "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014.dat"
MFile  =  "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014.rho"
MPad=[14, 14 , 14, 14, 0, 71]

start = time.time()
dx, dy, dz, rho, refmod, _, vcell = mod.read_model_mod(MFile, trans="linear", volumes=True)
dims = np.shape(rho)
sdims = np.size(rho)

elapsed = time.time() - start
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


rhoair = 1.e17
aircells = np.where(rho>rhoair/10)
blank = rhoair

jacmask = jac.set_mask(rho=rho, pad=MPad, blank= blank, flat = False, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)

TSTFile = "Test_modem"

mod.write_model_mod(TSTFile, dx, dy, dz, rho, reference=refmod, trans="LINEAR", mvalair=blank, aircells=aircells)

lat, lon =  -16.346,  -70.908
elev = -refmod[2]

refcenter =  [lat, lon, elev]
mod.write_model_ubc(TSTFile, dx, dy, dz, rho, refcenter, mvalair=blank, aircells=aircells)

start = time.time()
dxu, dyu, dzu, valu, refubc, _, vcell = mod.read_model_ubc(TSTFile, volumes=True, out=True)
elapsed = time.time() - start
print(" Used %7.4f s for reading model from %s " % (elapsed, TSTFile))

print(refmod)
print(refubc)
print(refmod-refubc[0:3])
