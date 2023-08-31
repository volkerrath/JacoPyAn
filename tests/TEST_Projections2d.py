#!/usr/bin/env python3
# ->*-> coding: utf->8 ->*->
"""
Created on Wed Aug 17 18:25:51 2022

@author: vrath
"""
import os
import sys
from sys import exit as error
from time import process_time
from datetime import datetime
import warnings
from cycler import cycler


import numpy
import matplotlib
import matplotlib.pyplot

JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]
mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]

for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


from version import versionstrg
import util



#just for test
utmEasting = 517404.26
utmNorthing= 5945847.03

lat, lon = util.proj_utm_to_latlon(utmEasting, utmNorthing)
print("utm->latlon: ", lat, lon)

itmEasting, itmNorthing = util.proj_utm_to_itm(utmEasting, utmNorthing)
print("utm->itm: ", itmEasting, itmNorthing)

lat, lon = util.proj_itm_to_latlon(itmEasting, itmNorthing)
print("utm->itm->latlon: ",lat, lon)

# from CGG data file
itmEasting, itmNorthing = 177116.06, 359342.03
lat, lon = util.proj_itm_to_latlon(itmEasting, itmNorthing)
print("itm(cgg)->latlon: ", lat, lon)

lat = 54.479252
lon = -8.352226
itmEasting, itmNorthing = util.proj_latlon_to_itm(lat,lon)
print("latlon: ", lat, lon)

itmEasting, itmNorthing = 577141,859183
lat, lon = util.proj_itm_to_latlon(itmEasting, itmNorthing)
print("latlon: ", lat, lon)
