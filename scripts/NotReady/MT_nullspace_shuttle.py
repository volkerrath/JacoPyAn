#!/usr/bin/env python3
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
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules

import os
import sys

# import struct
import time
from datetime import datetime
import warnings
import gc


import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
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
import util as utl

from version import versionstrg

gc.enable()


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

rng = np.random.default_rng()
nan = np.nan

# KRAFLA case
# WorkDir = "/media/vrath/BlackOne/MT_Data/Krafla/Krafla1/"
# MFile   = WorkDir +r"Krafla.rho"

# Annecy case
WorkDir = "/home/vrath/MT_Data/Annecy/Jacobians/"
MODFile = WorkDir+"ANN_best.rho"
SVDFile = WorkDir +"/svd/ANN_ZPT_nerr_sp-8_SVD_k500_o2_s0_96.8percent.npz"


total = 0.0
start =time.perf_counter()
print("\nReading Jacobina decomposition from "+SVDFile)

U = np.load(SVDFile)["U"]


elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading from %s " % (elapsed, SVDFile))


