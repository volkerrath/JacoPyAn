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
Reads ModEM model, reads ModEM"s Jacobian, does fancy things.

Created on Sun Jan 17 15:09:34 2021

@author: vrath jan 2021

"""

# Import required modules

import os
import sys
from sys import exit as error
# import struct
import time
from datetime import datetime

import numpy as np
import netCDF4 as nc

JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod



import np as np
import math as ma
import netCDF4 as nc

import modem as mod
import util as utl

from version import versionstrg

rng = np.random.default_rng()

nan = np.nan  # float("NaN")
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


rhoair = 1.e+17

total = 0
ModDir_in = JACOPYAN_DATA + "/Peru/Misti/"
ModDir_out = ModDir_in + "/results_shuttle/"

ModFile_in = JACOPYAN_DATA + "Misti10_best"    
ModFile_out = ModFile_in
   
ModOrig = [-16.277300, -71.444397]# Misti
# JacName = "Misti_best_Z5_nerr_sp-8"
JacFile = "Misti_best_ZT_extended_nerr_sp-8"


if not os.path.isdir(ModDir_out):
    print("File: %s does not exist, but will be created" % ModDir_out)
    os.mkdir(ModDir_out)


geocenter = [45.938251, 6.084900]
utm_x, utm_y = utl.project_latlon_to_utm(geocenter[0], geocenter[1], utm_zone=32631)
utmcenter = [utm_x, utm_y, 0.]

samples = 10000
smoother = None  # ["uniform", 3] ["gaussian",0.5]

#            geo    act   dlog10rho  position         axes                 angles
basebody = ["ell", "add", 0.2,      0., 0., 0.,   3000., 1000., 2000.,    0., 0., 30.]

total = 0.
start = time.perf_counter()

smoother = None # ["uniform", 1] ['gaussian',0.5]
total = 0
start = time.perf_counter()


dx, dy, dz, rho, refmod, _ = mod.read_mod(ModFile_in, ".rho",trans="log10", volumes=True)
aircells = np.where(rho>rhoair/10)

elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile_in + ".rho"))


dx, dy, dz, rho, reference = mod.read_mod(ModFile)
elapsed = (time.perf_counter() - start)
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, DatFile))


total = 0
start = time.perf_counter()

dx, dy, dz, rho, reference = mod.read_mod(ModFile_in, out=True)
elapsed = (time.perf_counter() - start)
total = total + elapsed
print(" Used %7.4f s for reading model from %s " %
      (elapsed, ModFile_in + ".rho"))

aircells = rho > rhoair / 100.

rho = mod.prepare_model(rho, rhoair=rhoair)

for ibody in range(samples):
    body = basebody.copy()

    rhonew = mod.insert_body(dx, dy, dz, rho, body)
    rhonew[aircells] = rhoair
    Modout = ModFile_out + "_" + body[0] + \
        str(ibody) + "_" + smoother[0] + ".rho"
    write_model_mod(Modout, dx, dy, dz, rhonew, reference, out=True)

    elapsed = (time.perf_counter() - start)
    print(
        " Used %7.4f s for processing/writing model to %s " %
        (elapsed, Modout))
    print("\n")


total = total + elapsed
print(" Total time used:  %f s " % (total))
