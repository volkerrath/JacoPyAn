#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:34:38 2024

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
import gc

import numpy as np
import numpy.linalg as npl
import scipy.linalg as scl
import scipy.sparse as scs
import netCDF4 as nc

import pyevtk.hl as vtx
# from pyevtk.hl import rectilinearToVTK, pointsToVTK, pointsToVTKAsTIN


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
blank = 1.e-30 # np.nan
rhoair = 1.e17

modExt = ".rho"

# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
# WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"
WorkDir = "/home/vrath/UBI38_JAC/"
ModFiles = ["Ubi38_ZssPT_Alpha02_NLCG_023"]
DatFiles = ["Ubi38_ZssPT_Alpha02_NLCG_023"]
SnsFiles = []

nfiles = len(ModFiles)

fcount =0
for ifile in np.arange(nfiles):
    infile = WorkDir+ModFiles[ifile]
    outfile = infile+".vtk"


    dx, dy, dz, rho, refmod, _ = mod.read_mod(infile, ".rho",trans="linear", volumes=True)
    
    aircells = np.where(rho>rhoair/10)
    rho = np.log10(rho)
    xm = utl.set_mesh(dx=dx, center=True)   
    ym = utl.set_mesh(dy=dy, center=True) 
    zm = - utl.set_mesh(dz=dz, center=False) 
    
    sites = np.array([])    
    
    comments = [ "", "" ]
    vtx.rectilinearToVTK(outfile, xm, ym, zm, cellData = {"rho" : rho})
    
    
    # rectilinearToVTK(outfile, xs, ys, zs, pointData = {"sites" : sites})

    

