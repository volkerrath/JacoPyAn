#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Read ModEM model, ModEM's Jacobian.

Model order reduction by discrete (co)sinus transform.

@author: vrath Feb 2021

"""
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



rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

