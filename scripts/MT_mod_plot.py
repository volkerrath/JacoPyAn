#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Mar 23, 2024

@author: vrath
"""
# Import required modules

import os
import sys
from sys import exit as error
# import struct
import time
from datetime import datetime
import warnings

import numpy as np
import numpy.linalg as npl

import scipy as sc
import scipy.linalg as scl
import scipy.sparse as scs

from osgeo import gdal

import discretize
import tarfile



import matplotlib as mpl
import matplotlib.pyplot as plt




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



rng = np.random.default_rng()
blank =  np.nan
rhoair = 1.e17
rhoair = np.log10(rhoair)

# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"
# WorkDir = "/home/vrath/UBI38_JAC/"

UseSens = True 
ModFile = "Ubi38_ZssPT_Alpha02_NLCG_023"
SnsFile = "/sens_euc/Ubi38_ZPT_nerr_sp-8_total_euc_sqr_max_sns"

Scale = 1.e-3  #  = km,  1 = m
Slices = [("xz", 0.), ("xz", -8.), ("xz", 8.),
          ("yz", 0.), ("yz", -8.), ("yz", 8.),
          ("xy", 0.), ("xy", 4.), ("xy", -9.),]
DistMask = [-15., 15.,   -15., 15.,  0., 25.]
mask = DistMask

infile = WorkDir+ModFile
dx, dy, dz, rho, refmod, _ = mod.read_mod(infile, ".rho",trans="log10") 
aircells = np.where(rho>rhoair-1.0)

if UseSens:
    infile = WorkDir+SnsFile
    _, _, _, sns, _, _ = mod.read_mod(infile, ".rho",trans="log10") 


msh = np.shape(rho)

x0, y0, z0 = mod.cells3d(dx, dy, dz) 
x0 =  (x0 + refmod[0])*Scale
y0 =  (y0 + refmod[1])*Scale
z0 =  (z0 + refmod[2])*Scale

# cell centers
xc = 0.5 * (x0[0:msh[0]] + x0[1:msh[0]+1])
yc = 0.5 * (y0[0:msh[1]] + y0[1:msh[1]+1])
zc = 0.5 * (z0[0:msh[2]] + z0[1:msh[2]+1])

x, y, z, rho = mod.mask_mesh(x0, y0, z0, mod=rho, mask=mask, method="dist")
if UseSens: 
    _, _, _, sns = mod.mask_mesh(x, y, z, mod=rho, mask=mask, method="dist")
    
nslices = np.shape(Slices)[0]
for sl in np.arange(nslices):
    s = Slices[sl]
    print("\nSlice "+str(sl))
    print("Direction: "+s[0])
    print("Position: "+str(s[1]))
    z = -z
    pos = s[1]
    if "xz" in s[0]:
        nearest = np.array(pos - yc).argmin()
        slicr = rho[nearest, :, :]
        if UseSens:
            slics = sns[nearest, :, :]
            
        X, Y = np.meshgrid(y, z)
                
    if "yz" in s[0]:
        nearest = np.array(pos - xc).argmin()
        slicr = rho[:, nearest, :]
        if UseSens:
            slics = sns[:, nearest, :]
            
            X, Y = np.meshgrid(x, z)
        
    if "xy" in s[0]:
        nearest = np.array(pos - zc.argmin()
        slicr = rho[:, :, nearest]
        if UseSens:
            slics = sns[:, nearest, :]