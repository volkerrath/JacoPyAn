#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:34:38 2024

@author: vrath
"""
import numpy as np
import os
from evtk.hl import rectilinearToVTK, pointsToVTK, pointsToVTKAsTIN

print("Running rectilinear...")

# Dimensions
nx, ny, nz = 6, 6, 2
lx, ly, lz = 1.0, 1.0, 1.0
dx, dy, dz = lx/nx, ly/ny, lz/nz

ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Coordinates
x = np.arange(0, lx + 0.1*dx, dx, dtype='float64')
y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')
z = np.arange(0, lz + 0.1*dz, dz, dtype='float64')

# Variables
pressure = np.random.rand(ncells).reshape( (nx, ny, nz))
temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1))

comments = [ "comment 1", "comment 2" ]
rectilinearToVTK(FILE_PATH, x, y, z, cellData = {"pressure" : pressure}, pointData = {"temp" : temp}, comments = comments)


print("Running points...")

# Example 1: Random points
npoints = 100
x = np.random.rand(npoints)
y = np.random.rand(npoints)
z = np.random.rand(npoints)
pressure = np.random.rand(npoints)
temp = np.random.rand(npoints)
comments = [ "comment 1", "comment 2" ]

# keys are sorted before exporting, hence it is useful to prefix a number to determine an order
pointsToVTK(FILE_PATH1, x, y, z, data = {"1_temp" : temp, "2_pressure" : pressure}, comments = comments) 

# Example 2: Export as TIN
ndim = 2 #only consider x, y coordinates to create the triangulation
pointsToVTKAsTIN(FILE_PATH2, x, y, z, ndim = ndim, data = {"1_temp" : temp, "2_pressure" : pressure}, comments = comments)

# Example 3: Regular point set
x = np.arange(1.0,10.0,0.1)
y = np.arange(1.0,10.0,0.1)
z = np.arange(1.0,10.0,0.1)

comments = [ "comment 1", "comment 2" ]
pointsToVTK(FILE_PATH3, x, y, z, data = {"elev" : z}, comments = comments)

# Example 4: Point set of 5 points
x = [0.0, 1.0, 0.5, 0.368, 0.4]
y = [0.3, 2.0, 0.7, 0.1, 0.6]
z = [1.0, 1.0, 0.3, 0.75, 0.9]
pressure = [1.0, 2.0, 3.0, 4.0, 5.0]
temp = [1.0, 2.0, 3.0, 4.0, 5.0]
comments = [ "comment 1", "comment 2" ]

# keys are sorted before exporting, hence it is useful to prefix a number to determine an order
pointsToVTK(FILE_PATH4, x, y, z, data = {"1_temp" : temp, "2_pressure" : pressure}, comments = comments) 
