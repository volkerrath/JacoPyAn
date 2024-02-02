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
Reads ModEM'smodel file, adds perturbations.

Ellipsoids or boxes

@author: vr jan 2021


"""

import os
import sys
from sys import exit as error
import time
from datetime import datetime

import numpy as np

#import netCDF4 as nc
#import scipy.ndimage as spn
#import scipy.linalg as spl

#import vtk
#import pyvista as pv
#import PVGeo as pvg
#import omf
#import omfvista as ov
#import gdal


JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import modem as mod
import util as utl
import jacproc as jac
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

rhoair = 1.e17

ModFile_in = JACOPYAN_DATA +"/1_feb_ell/TAC_100"
ModFile_out = ModFile_in

geocenter = [-17.489, -70.031]
utm_x, utm_y = utl.proj_latlon_to_utm(geocenter[0], geocenter[1], utm_zone=32719)
utmcenter = [utm_x, utm_y, 0.0]

ell = ["ell", "rep", 10000., 0., 0., 10000., 30000., 30000., 2000., 0., 0.,0.]
#box = ["box", "rep", 10., 0., 0., 0., 1000., 2000., 1000., 1000., 0., 0., 30.]
#cyl = []

bodies = [ell]
nb = np.shape(bodies)
# smoother=['gaussian',0.5]
smoother = ["uniform", 1]
total = 0
start = time.perf_counter()


dx, dy, dz, rho, refmod, _ = mod.read_mod(ModFile_in, ".rho",trans="linear", volumes=True)
aircells = np.where(rho>rhoair/10)

elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile_in + ".rho"))


rho = mod.prepare_model(rho, rhoair=rhoair)

for ibody in range(nb[0]):
    body = bodies[ibody]
    rhonew = mod.insert_body(dx, dy, dz, rho, body, smooth=smoother, reference= refmod)
    #rhonew[air] = rhoair
    Modout = ModFile_out+"_"+body[0]+str(ibody)+"_"+smoother[0]
    mod.write_mod(Modout, modext="_new.rho",trans = "LOGE",
                  dx=dx, dy=dy, dz=dz, mval=rhonew,
                  reference=refmod, mvalair=1E+17, aircells=aircells, header="")

    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for processing/writing model to %s"
          % (elapsed, Modout))
    print("\n")


total = total + elapsed
print(" Total time used:  %f s " % (total))