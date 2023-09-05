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
Read ModEM model, ModEM's Jacobian.

Model order reduction by discrete Hermite transform.

@author: vrath Feb 2021

"""

# Import required modules

import os
import sys
from sys import exit as error
import time
import warnings

import numpy as np
import math as ma
import netCDF4 as nc
import scipy.ndimage as spn
import scipy.linalg as spl

import vtk
import pyvista as pv
import PVGeo as pvg

JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]
mypath = [JACOPYAN_ROOT+"/JacoPyAn/modules/", JACOPYAN_ROOT+"/JacoPyAn/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.append(pth)

import modules
import modem as mod
import util as utl


from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

nan = np.nan  # float("NaN")
rng = np.random.default_rng()
blank = np.nan
rhoair = 1.e17


trans = "LINEAR"
InpFormat = "sparse"
OutFormat = "mod ubc"
# UBINAS
# WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/UbiJac/"
WorkDir = JACOPYAN_ROOT+"/work/"
WorkName = "UBI_ZPT_nerr_sp-8"
MFile   = WorkDir + "UBI_best.rho"
