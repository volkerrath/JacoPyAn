#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: "1.5"
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
from pyvista import themes
from pyvistaqt import BackgroundPlotter
import discretize
import tarfile
import pylab as pl


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

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)



Task = "ortho"
Task = "slices"
#Task = "con"
# Task = "lines"
#

Bounds = [-5.,5., -5.,5., -1. ,3.]
Pads = [13, 13, 40]
Scale = 1.e-3
LimLogRes = [-1, 4.]
StepContrs=0.5
UseSens  = True 
MinLOgSns = -4.
UseData = False

Cmap = "jet"

if "ortho" in Task.lower():
    posx, posy, posz = 0., 0., -8.


# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
Title = "Ubinas Volcano, Peru"
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"
if not WorkDir.endswith("/"):
    WorkDir = WorkDir+"/"
Datfile = WorkDir + "U38_ZPTss1"
ModFile = WorkDir + "Ubi38_ZssPT_Alpha02_NLCG_023"
SnsFile = WorkDir+"/sens_euc/Ubi38_ZPT_nerr_sp-8_total_euc_sqr_max_sns"

Plotfile = WorkDir + "Ubi38_ZssPT_Alpha02_NLCG_023"



dx, dy, dz, rho, reference, _ = mod.read_mod(ModFile, ".rho",trans="log10")
aircells = np.where(rho>15.)
rho[aircells]=np.NaN
print("Reading model from %s " % (ModFile))
print("ModEM reference is "+str(reference))
print("Min/max rho = "+str(np.nanmin(rho))+"/"+str(np.nanmax(rho)))
x, y, z, vals = mod.model_to_pv(dx=dx, dy=dy, dz=dz, rho=rho, 
                                reference=reference, scale=Scale, pad=Pads)

if UseSens:
    dx, dy, dz, sns, reference, _ = mod.read_mod(SnsFile,".rho",trans="log10")
    print("Reading senitivity from %s " % (SnsFile))
    _, _, _, sens = mod.model_to_pv(dx=dx, dy=dy, dz=dz, rho=rho, 
                                    reference=reference, scale=Scale, pad=Pads)

if UseData:
    
    site, _, data, _ = mod.read_data(Datfile=Datfile)


    xdat, ydat, z, sites, siten = mod.data_to_pv(data=data, site=site, 
                                                reference=reference, scale=1.)

# comments = [ "", "" ]
# f= vtx.pointsToVTK(outfile, x, y, z, data = {"sites" : sites})
# print("sites written to ", f)   

cmap = mpl.colormaps[Cmap]

pv.set_jupyter_backend("trame")
mtheme = pv.themes.DocumentTheme()
mtheme.nan_color = "white"
mtheme.above_range_color ="white"
# my_theme.lighting = False
# my_theme.show_edges = True
# my_theme.axes.box = True

# mtheme.colorbar_orientation = "vertical"
# mtheme.colorbar_vertical.height=0.6
# mtheme.colorbar_vertical.position_y=0.2
# mtheme.colorbar_vertical.title="log resistivity"
mtheme.font.size=14
mtheme.font.title_size=16
mtheme.font.label_size=14


pv.global_theme.load_theme(mtheme)


lut = pv.LookupTable()
lut.apply_cmap(cmap, n_values=128, flip=True)
lut.scalar_range = LimLogRes
lut.above_range_color = None
lut.nan_color = None




model = pv.RectilinearGrid(y, x, z)

# RectilinearGrid.cast_to_structured_grid()
model.cell_data["resistivity"] = vals

# contours = mod.contour(np.linspace(1.5, 2.5, 6))


p = pv.Plotter(window_size=[1600, 1300], theme=mtheme, notebook=False, off_screen=False)
p.disable_shadows()
# p.viewport = (0.05, 0.05, 0.95, 0.95)
p.add_title(Title)
_ = p.add_mesh(model.outline(), color="k")
grid_labels = dict(ztitle="elev (km)", xtitle="w-e (km)", ytitle="s-n (km)")
p.show_grid(**grid_labels)

if "ortho" in Task.lower():
    slicepars = dict(clim=LimLogRes, 
                     cmap=lut,
                     above_color=None, 
                     nan_color="white",
                     nan_opacity=1.,
                     show_scalar_bar=False,
                     interpolate_before_map=True,
                     log_scale=False)
    slices = model.slice_orthogonal(x=posx, y=posy, z=posz)
    _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
    
    
elif"slice" in Task.lower():
    slicepars = dict(clim=LimLogRes, 
                 cmap=lut,
                 above_color=None, 
                 nan_color="white",
                 nan_opacity=1.,
                 opacity=1.,
                 interpolate_before_map=True,
                 show_scalar_bar=False,
                 log_scale=False)
    # slices = model.slice(x=0., y=0., z=-5.)
    slices= model.slice(normal=[1, 1, 0])
    _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
    
    
elif"spread" in Task.lower():
    slices = model.slice_orthogonal(x=0., y=0., z=-5.)
    _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
      
elif "cont" in Task.lower():
    
    cntrs = np.arange(LimLogRes[0], LimLogRes[1],0.5)
    modiso = model.contour(cntrs, scalars="resistivity")
    _ = p.add_mesh(modiso, scalars="resistivity", **slicepars)

elif "line" in Task.lower():
    slices = model.slice_orthogonal(x=0., y=0., z=-5.)
    
    
_ = p.add_scalar_bar(title="log res", 
                         vertical=True, 
                         position_y=0.2,
                         position_x=0.9,
                         height=0.6,
                         title_font_size=26, 
                         label_font_size=16,
                         bold=False,
                         n_labels = 6,
                         fmt='%3.1f')


p.add_axes()
p.save_graphic("img.pdf")
p.show()
p.save_graphic("img2.pdf") 
p.close()
 
plt.box(False)
plt.imshow(p.image)

fig = plt.imshow(p.image)
fig. frameon=False
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig("test.pdf", dpi=600, edgecolor=None)
plt.savefig("test.png", dpi=600, transparent=True, edgecolor=None)
plt.savefig("test.svg", dpi=600, transparent=True, edgecolor=None)
