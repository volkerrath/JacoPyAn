#!/usr/bin/env python3
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

import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np
from osgeo import gdal
import scipy as sc
import vtk
import pyvista as pv
# import pyvistaqt as pvqt
import discretize
import tarfile
import pylab as pl
from time import sleep

import matplotlib as mpl
import matplotlib.pyplot as plt


JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

rng = np.random.default_rng()
blank = 1.e-30 # np.nan
rhoair = 1.e17
nan = np.nan  # float("NaN")


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)


Bounds = [-5.,5., -5.,5., -1. ,3.]
Pads = [12, 12, 36]
Scale = 1.e-3
Center = False
MaxLogRes = 3.
MinLOgSns = -4.
Cmap = "viridis"



# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"
# WorkDir = "/home/vrath/UBI38_JAC/"
ModFile = WorkDir + "Ubi38_ZssPT_Alpha02_NLCG_023"
SnsFile = WorkDir+"/sens_euc/Ubi38_ZPT_nerr_sp-8_total_euc_sqr_max_sns"



# WorkDir = JACOPYAN_DATA+"Annecy/Jacobians/"
# if not WorkDir.endswith("/"):
#     WorkDir = WorkDir+"/" 
# ModFile = WorkDir+"ANN_best"
# SnsFile = WorkDir+"/sens_euc/"

total = 0
start = time.perf_counter()
dx, dy, dz, rho, reference, _ = mod.read_mod(ModFile, ".rho",trans="log10")

elapsed = time.perf_counter() - start
total = total + elapsed
print("Used %7.4f s for reading model from %s " % (elapsed, ModFile))
print("ModEM reference is "+str(reference))
print("Min/max rho = "+str(np.min(rho))+"/"+str(np.max(rho)))

# start = time.perf_counter()
# dx, dy, dz, sns, reference, _ = mod.read_mod(SnsFile,".rho",trans="log10")
# elapsed = time.perf_counter() - start
# total = total + elapsed
# print("Used %7.4f s for reading model from %s " % (elapsed, SnsFile))


x, y, z = mod.cells3d(dx, dy, dz)
z = -z

if Center: 
    x = x - 0.5*(x[-1]-x[0])
    y = y - 0.5*(y[-1]-y[0])
else:
    x =  x + reference[0]
    y =  y + reference[1]
    
if Scale !=None:
    x, y, z  = Scale*x, Scale*y, Scale*z


   
rho[rho>MaxLogRes] = np.nan 
x, y, z, rho = mod.clip_model(x, y, z, rho, pad = Pads)
vals = np.swapaxes(np.flip(rho, 2), 0, 1).flatten(order="F")




cmap = mpl.colormaps[Cmap]
dargs = dict(cmap=cmap, clim=[-1., 3.])

pv.set_plot_theme("document")
model = pv.RectilinearGrid(y, x, -z)

# RectilinearGrid.cast_to_structured_grid()
model.cell_data["resistivity"] = vals

# contours = mod.contour(np.linspace(1.5, 2.5, 6))

p = pv.Plotter(window_size=[2*1024, 2*768])
_ = p.add_mesh(model.outline(), color="k")
# _ = p.add_mesh(contours, opacity=0.25, clim=[1.4, 2.6])
p.show_grid()
slices = model.slice_orthogonal(x=0., y=0, z=-3.)
# _ = p.add_mesh(mod, scalars="resistivity")
_ = p.add_mesh(slices, scalars="resistivity", **dargs)
p.add_title("Annecy")
p.add_axes()
p.show(screenshot='my_image.png',auto_close=True)
p.close()

# slices = mod.slice_orthogonal()

# slices.plot(cmap=cmap)
