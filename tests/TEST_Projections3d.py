#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:05:26 2021

@author: vrath
"""



import os
import sys
from sys import exit as error
from time import process_time
from datetime import datetime
import warnings

import numpy as np
import pyproj as proj

JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]
mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

from version import versionstrg
import util as utl




"""
Coordinate transformations using pyproj
Look for other EPSG at https://epsg.io/
VR 08/21

EPSG (1D):
    5714    (Geoid height, masl)
    5715    (Geoid depth,mbsl)
    3859    (EGM2008 Geoid height,masl)
    224     (WGS84 ellipsoid height)

EPSG (2D):
    32629   (2D, UTM 29N,   E, N)
    4326    (2D, WGS84,     Lat, Lon)
    2157    (2D, Irenet95,  E, N,)

EPSG (3D):
    4979    (3D, WGS84,     Lat, Lon, h)
    4351    (3D, Irenet,     E, N, h)


EPSG:8403 or 1 	ETRF2014	Geographic 3D

"""

A = np.load("Testdata.npz")

Data = A["TestData"]
print(np.shape(Data))


inp_x = Data[:, 1]
inp_y = Data[:, 2]
inp_h = Data[:, 3]

InProj =  32629
# OutProj = 4979
# t3d = proj.Transformer.from_crs(
#             proj.CRS(32629).to_3d(),
#             proj.CRS(4326).to_3d())
t3d = proj.Transformer.from_crs(
            proj.CRS(32629).to_3d(),
            proj.CRS(4979),always_xy=True)
# t3d = proj.Transformer.from_crs(
#             proj.CRS(32629).to_3d(),
#             OutProj)
# t2d = proj.Transformer.from_crs(InProj, OutProj)
lon, lat, alt = t3d.transform(inp_x, inp_y, inp_h)


outproj1 =proj.crs.CompoundCRS(name="WGS 84 + EGM2008 height", components=[4326, 3855])
outproj2 =proj.crs.CompoundCRS(name="WGS 84 + EGM2008 height", components=[32629, 3855])

t3d = proj.Transformer.from_crs(
            proj.CRS(4979),
            outproj1, always_xy=True)
out1_x, out1_y, out1_h = t3d.transform(lat, lon, alt)

t3d = proj.Transformer.from_crs(
            proj.CRS(32629),
            outproj2, always_xy=True)
out2_x, out2_y, out2_h = t3d.transform(inp_x, inp_y, inp_h)
