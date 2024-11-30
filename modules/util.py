"""
Created on Sun Nov  1 17:08:06 2020

@author: vrath
"""

import os
import sys
import ast
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, convolve, gaussian_gradient_magnitude
from scipy.linalg import norm
from sys import exit as error
import fnmatch
# from numba import jit
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import pyproj
from pyproj import CRS, Transformer


from sys import exit as error



from scipy.fftpack import dct, idct

#def error(out_string="Unknown error occured! Exit.")
        #exit(out_string)

def warn(out_string=None):
        if out_string is not None:
            print("Warning: "+out_string)

def dctn(x, mnorm="ortho"):
    """
    Discrete cosine transform (fwd)
    https://stackoverflow.com/questions/13904851/use-pythons-scipy-dct-ii-to-do-2d-or-nd-dct
    """
    for i in range(x.ndim):
        x = dct(x, axis=i, norm=mnorm)
    return x

def idctn(x, mnorm="ortho"):
    """
    Discrete cosine transform (inv)
    https://stackoverflow.com/questions/13904851/use-pythons-scipy-dct-ii-to-do-2d-or-nd-dct
    """
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=mnorm)
    return x

def indexrange(bb, bl):
    """
    From GIBBON toolbox (https://www.gibboncode.org/):

        function i=indexRange(blockSize,s)

        i=repmat(1:blockSize:s,blockSize,1);
        i=i(:);
        if numel(i)<s
            ii=i;
            i=siz(1)*ones(1,s);
            i(1:num_i)=ii;
        else
            i=i(1:s);
        end
        [~,~,i]=unique(i);
        end
    """
    if iseven(bl):
        bl=bl+1
        print("block size was even, increased to ", bl)

    if bb <= bl:
        error("bounding box too small!Exit.")


    # import numpy.matlib
    # jrange = numpy.matlib.repmat(np.arange(0,bbsize,block),block,1)

    irange = np.tile(np.arange(0,bb,bl),(bl,1))

    return irange

def iseven(num):
    even = False
    if num % 2 == 0:
        even= True

    return even

def wait1d(periods=np.array([]),
           thick=np.array([]), cond=np.array([]), out=False):

    """
    Calculate magnetotelluric impedance Z , apparent resitivity rhoa, and phase phi for a given layer model.

    """

    scale = 1 / (4 * np.pi / 10000000)
    mu = 4 * np.pi * 1e-7 * scale
    omega = 2 * np.pi / periods

    Z = np.zeros(len(periods), dtype=complex)
    rhoa = np.zeros(len(periods))
    phi = np.zeros(len(periods))

    for nfreq, w in enumerate(omega):
        prop_const = np.sqrt(1j*mu*cond[-1] * w)
        C = np.zeros(len(rhoa), dtype=complex)
        C[-1] = 1./prop_const
        if len(thick) > 1:
            for k in reversed(range(len(rhoa) - 1)):
                prop_layer = np.sqrt(1j*w*mu*cond[k])
                k1 = (C[k+1] * prop_layer + np.tanh(prop_layer * thick[k]))
                k2 = ((C[k+1] * prop_layer * np.tanh(prop_layer * thick[k])) + 1)
                C[k] = (1./ prop_layer) * (k1 / k2)

        Z[nfreq] = 1j * w * mu * C[0]

    rhoa = 1./omega*np.abs(Z)**2;
    phi = np.angle(Z, deg=True)

    return Z, rhoa, phi

def parse_ast(filename):
    with open(filename, "rt") as file:

        return ast.parse(file.read(), filename=filename)

def check_env(envar="CONDA_PREFIX", action="error"):
    """
    Check if environment variable exists

    Parameters
    ----------
    envar : strng, optional
        The default is ["CONDA_PREFIX"].

    Returns
    -------
    None.

    """
    act_env = os.environ[envar]
    if len(act_env)>0:
        print("\n\n")
        print("Active conda Environment  is:  " + act_env)
        print("\n\n")
    else:
        if "err" in action.lower():
            error("Environment "+ act_env+"is not activated! Exit.")


def find_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))


def list_functions(filename):
    """
    Generate list of functions in module.

    author: VR 3/21
    """

    print(filename)
    tree = parse_ast(filename)
    for func in find_functions(tree.body):
        print("  %s" % func.name)


def get_filelist(searchstr=["*"], searchpath="."):
    """
    Generate filelist from path and unix wildcard list.

    author: VR 3/20
    """

    filelist = fnmatch.filter(os.listdir(searchpath), "*")
    for sstr in searchstr:
        filelist = fnmatch.filter(filelist, sstr)
        print(filelist)

    return filelist

def get_utm_zone(latitude=None, longitude=None):
    """
    Find EPSG from position, using pyproj

    VR 08/23
    """
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info
    utm_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
        west_lon_degree=longitude,
        south_lat_degree=latitude,
        east_lon_degree=longitude,
        north_lat_degree=latitude, ), )
    utm_crs =CRS.from_epsg(utm_list[0].code)
    EPSG = CRS.to_epsg(utm_crs)

    return EPSG, utm_crs


def proj_latlon_to_utm(latitude, longitude, utm_zone=32629):
    """
    transform latlon to utm , using pyproj
    Look for other EPSG at https://epsg.io/

    VR 08/23
    """
    prj_wgs = CRS("epsg:4326")
    prj_utm = CRS("epsg:" + str(utm_zone))
    transformer = Transformer.from_crs(prj_wgs, prj_utm)
    utm_x, utm_y = transformer.transform(latitude, longitude)
    # transfor
    # prj_wgs = CRS("epsg:4326")
    # prj_utm = CRS("epsg:" + str(utm_zone))
    # utm_x, utm_y = pyproj.transform(prj_wgs, prj_utm, latitude, longitude)

    return utm_x, utm_y

def proj_utm_to_latlon(utm_e, utm_n, utm_zone=32629):
    """
    transform latlon to utm , using pyproj
    Look for other EPSG at https://epsg.io/

    VR 08/23
    """
    prj_wgs = CRS("epsg:4326")
    prj_utm = CRS("epsg:" + str(utm_zone))
    transformer = Transformer.from_crs(prj_utm, prj_wgs)
    latitude, longitude = transformer.transform(utm_e, utm_n)

    return latitude, longitude


def proj_latlon_to_itm(longitude, latitude):
    """
    transform latlon to itm , using pyproj
    Look for other EPSG at https://epsg.io/

    VR 08/23
    """
    prj_wgs = CRS("epsg:4326")
    prj_itm = CRS("epsg:2157")
    transformer = Transformer.from_crs(prj_wgs, prj_itm)
    itm_e, itm_n = transformer.transform(latitude, longitude)

    return itm_e, itm_n


def proj_itm_to_latlon(itm_e, itm_n):
    """
    transform itm to latlon, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 08/23
    """
    prj_wgs = CRS("epsg:4326")
    prj_itm = CRS("epsg:2157")
    transformer = Transformer.from_crs(prj_itm, prj_wgs)
    latitude, longitude = transformer.transform(itm_e, itm_n)

    return latitude, longitude


def proj_itm_to_utm(itm_x, itm_y, utm_zone=32629):
    """
    transform itm to utm, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 08/23
    """
    prj_utm = CRS("epsg:" + str(utm_zone))
    prj_itm = CRS("epsg:2157")
    transformer = Transformer.from_crs(prj_itm, prj_utm)
    utm_e, utm_n = transformer.transform(itm_x, itm_y)

    return utm_e, utm_n


def proj_utm_to_itm(utm_e, utm_n, utm_zone=32629):
    """
    transform utm to itm, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 08/23
    """
    prj_utm = CRS("epsg:" + str(utm_zone))
    prj_itm = CRS("epsg:2157")

    transformer = Transformer.from_crs(prj_utm, prj_itm)
    itm_e, itm_n = transformer.transform(utm_e, utm_n)

    return itm_e, itm_n


def proj_utm_to_utm(utm_e_in, utm_n_in, utmz_in=32629, utmz_out=32629):
    """
    transform utm to utm, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 08/23

    """
    if utmz_in==utmz_out:
        return  utm_e_in, utm_n_in

    prj_utm_in = CRS("epsg:" + str(utmz_in))
    prj_utm_out = CRS("epsg:" + str(utmz_out))

    transformer = Transformer.from_crs(prj_utm_in, prj_utm_out)
    utm_e, utm_n = transformer.transform(utm_e_in, utm_n_in)

    return utm_e, utm_n


def project_wgs_to_geoid(lat, lon, alt, geoid=3855 ):
    """
    transform ellipsoid heigth to geoid, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 09/21

    """

    geoidtrans =pyproj.crs.CompoundCRS(name="WGS 84 + EGM2008 height", components=[4979, geoid])
    wgs = pyproj.Transformer.from_crs(
            pyproj.CRS(4979), geoidtrans, always_xy=True)
    lat, lon, elev = wgs.transform(lat, lon, alt)

    return lat, lon, elev

def project_utm_to_geoid(utm_x, utm_y, utm_z, utm_zone=32629, geoid=3855):
    """
    transform ellipsoid heigth to geoid, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 09/21

    """

    geoidtrans =pyproj.crs.CompoundCRS(name="UTM + EGM2008 height", components=[utm_zone, geoid])
    utm = pyproj.Transformer.from_crs(
            pyproj.CRS(utm_zone), geoidtrans, always_xy=True)
    utm_x, utm_y, elev = utm.transform(utm_x, utm_y, utm_z)

    return utm_x, utm_y, elev

def splitall(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_files(SearchString=None, SearchDirectory="."):
    """
    FileList = get_files(Filterstring) produces a list
    of files from a searchstring (allows wildcards)

    VR 11/20
    """
    FileList = fnmatch.filter(os.listdir(SearchDirectory), SearchString)

    return FileList



def unique(list, out=False):
    """
    find unique elements in list/array

    VR 9/20
    """

    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    if out:
        for x in unique_list:
            print(x)

    return unique_list


def strcount(keyword=None, fname=None):
    """
    count occurences of keyword in file
     Parameters
    ----------
    keywords : TYPE, optional
        DESCRIPTION. The default is None.
    fname : TYPE, optional
        DESCRIPTION. The default is None.

    VR 9/20
    """
    with open(fname, "r") as fin:
        return sum([1 for line in fin if keyword in line])
    # sum([1 for line in fin if keyword not in line])


def strdelete(keyword=None, fname_in=None, fname_out=None, out=True):
    """
    delete lines containing on of the keywords in list

    Parameters
    ----------
    keywords : TYPE, optional
        DESCRIPTION. The default is None.
    fname_in : TYPE, optional
        DESCRIPTION. The default is None.
    fname_out : TYPE, optional
        DESCRIPTION. The default is None.
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    VR 9/20
    """
    nn = strcount(keyword, fname_in)

    if out:
        print(str(nn) + " occurances of <" + keyword + "> in " + fname_in)

    # if fname_out is None: fname_out= fname_in
    with open(fname_in, "r") as fin, open(fname_out, "w") as fou:
        for line in fin:
            if keyword not in line:
                fou.write(line)


def strreplace(key_in=None, key_out=None, fname_in=None, fname_out=None):
    """
    replaces key_in in keywords by key_out

    Parameters
    ----------
    key_in : TYPE, optional
        DESCRIPTION. The default is None.
    key_out : TYPE, optional
        DESCRIPTION. The default is None.
    fname_in : TYPE, optional
        DESCRIPTION. The default is None.
    fname_out : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    VR 9/20

    """

    with open(fname_in, "r") as fin, open(fname_out, "w") as fou:
        for line in fin:
            fou.write(line.replace(key_in, key_out))


def gen_grid_latlon(
        LatLimits=None,
        nLat=None,
        LonLimits=None,
        nLon=None,
        out=True):
    """
     Generates equidistant 1-d grids in latLong.

     VR 11/20
    """
    small = 0.000001
# LonLimits = ( 6.275, 6.39)
# nLon = 31
    LonStep = (LonLimits[1] - LonLimits[0]) / nLon
    Lon = np.arange(LonLimits[0], LonLimits[1] + small, LonStep)

# LatLimits = (45.37,45.46)
# nLat = 31
    LatStep = (LatLimits[1] - LatLimits[0]) / nLat
    Lat = np.arange(LatLimits[0], LatLimits[1] + small, LatStep)

    return Lat, Lon


def gen_grid_utm(XLimits=None, nX=None, YLimits=None, nY=None, out=True):
    """
     Generates equidistant 1-d grids in m.

     VR 11/20
    """

    small = 0.000001
# LonLimits = ( 6.275, 6.39)
# nLon = 31
    XStep = (XLimits[1] - XLimits[0]) / nX
    X = np.arange(XLimits[0], XLimits[1] + small, XStep)

# LatLimits = (45.37,45.46)
# nLat = 31
    YStep = (YLimits[1] - YLimits[0]) / nY
    Y = np.arange(YLimits[0], YLimits[1] + small, YStep)

    return X, Y


def choose_data_poly(Data=None, PolyPoints=None, Out=True):
    """
     Chooses polygon area from aempy data set, given
     PolyPoints = [[X1 Y1,...[XN YN]]. First and last points will
     be connected for closure.

     VR 11/20
    """
    if Data.size == 0:
        error("No Data given!")
    if not PolyPoints:
        error("No Rectangle given!")

    Ddims = np.shape(Data)
    if Out:
        print("data matrix input: " + str(Ddims))

    Poly = []
    for row in np.arange(Ddims[0] - 1):
        if point_inside_polygon(Data[row, 1], Data[row, 1], PolyPoints):
            Poly.append(Data[row, :])

    Poly = np.asarray(Poly, dtype=float)
    if Out:
        Ddims = np.shape(Poly)
        print("data matrix output: " + str(Ddims))

    return Poly

# potentially faster:
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

# lons_lats_vect = np.column_stack((lons_vect, lats_vect)) # Reshape coordinates
# polygon = Polygon(lons_lats_vect) # create polygon
# point = Point(y,x) # create point
# print(polygon.contains(point)) # check if polygon contains point
# print(point.within(polygon)) # check if a point is in the polygon

# @jit(nopython=True)


def point_inside_polygon(x, y, poly):
    """
    Determine if a point is inside a given polygon or not, where
    the Polygon is given as a list of (x,y) pairs.
    Returns True  when point (x,y) ins inside polygon poly, False otherwise

    """
    # @jit(nopython=True)
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def choose_data_rect(Data=None, Corners=None, Out=True):
    """
     Chooses rectangular area from aempy data set, giveb
     the left lower and right uper corners in m as [minX maxX minY maxY]

    """
    if Data.size == 0:
        error("No Data given!")
    if not Corners:
        error("No Rectangle given!")

    Ddims = np.shape(Data)
    if Out:
        print("data matrix input: " + str(Ddims))
    Rect = []
    for row in np.arange(Ddims[0] - 1):
        if (Data[row, 1] > Corners[0] and Data[row, 1] < Corners[1] and
                Data[row, 2] > Corners[2] and Data[row, 2] < Corners[3]):
            Rect.append(Data[row, :])
    Rect = np.asarray(Rect, dtype=float)
    if Out:
        Ddims = np.shape(Rect)
        print("data matrix output: " + str(Ddims))

    return Rect


def proj_to_line(x, y, line):
    """
    Projects a point onto a line, where line is represented by two arbitrary
    points. as an array
    """
#    http://www.vcskicks.com/code-snippet/point-projection.php
#    private Point Project(Point line1, Point line2, Point toProject)
# {
#    double m = (double)(line2.Y - line1.Y) / (line2.X - line1.X);
#    double b = (double)line1.Y - (m * line1.X);
#
#    double x = (m * toProject.Y + toProject.X - m * b) / (m * m + 1);
#    double y = (m * m * toProject.Y + m * toProject.X + b) / (m * m + 1);
#
#    return new Point((int)x, (int)y);
# }
    x1 = line[0, 0]
    x2 = line[1, 0]
    y1 = line[0, 1]
    y2 = line[1, 1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)

    xn = (m * y + x - m * b) / (m * m + 1.)
    yn = (m * m * y + m * x + b) / (m * m + 1.)

    return xn, yn


def gen_searchgrid(Points=None,
                   XLimits=None, dX=None, YLimits=None, dY=None, Out=False):
    """
    Generate equidistant grid for searching (in m).

    VR 02/21
    """
    small = 0.1

    datax = Points[:, 0]
    datay = Points[:, 1]
    nD = np.shape(Points)[0]

    X = np.arange(np.min(XLimits), np.max(XLimits) + small, dX)
    nXc = np.shape(X)[0]-1
    Y = np.arange(np.min(YLimits), np.max(YLimits)+ small, dY)
    nYc = np.shape(Y)[0]-1
    if Out:
        print("Mesh size: "+str(nXc)+"X"+str(nYc)
              +"\nCell sizes: "+str(dX)+"X"+str(dY)
              +"\nNuber of data = "+str(nD))


    p = np.zeros((nXc, nYc), dtype=object)
    # print(np.shape(p))


    xp= np.digitize(datax, X, right=False)
    yp= np.digitize(datay, Y, right=False)

    for ix in np.arange (nXc):
        incol = np.where(xp == ix)[0]
        for iy in np.arange (nYc):
            rlist = incol[np.where(yp[incol] == iy)[0]]
            p[ix,iy]=rlist

    # pout = np.array(p,dtype=object)

            if Out:
               print("mesh cell: "+str(ix)+" "+str(iy))

    return p


def fractrans(m=None, x=None , a=0.5):
    """
    Caklculate fractional derivative of m.

    VR Apr 2021
    """
    import differint as df

    if m == None or x is None:
        error("No vector for diff given! Exit.")

    if np.size(m) != np.size(x):
        error("Vectors m and x have different length! Exit.")

    x0 = x[0]
    x1 = x[-1]
    npnts = np.size(x)
    mm = df.differint(a, m, x0, x1, npnts)

    return mm


def calc_resnorm(data_obs=None, data_calc=None, data_std=None, p=2):
    """
    Calculate the p-norm of the residuals.

    VR Jan 2021

    """
    if data_std is None:
        data_std = np.ones(np.shape(data_obs))

    resid = (data_obs - data_calc) / data_std

    rnormp = np.power(resid, p)
    rnorm = np.sum(rnormp)
    #    return {'rnorm':rnorm, 'resid':resid }
    return rnorm, resid


def calc_rms(dcalc=None, dobs=None, Wd=1.0):
    """
    Calculate the NRMS ans SRMS.

    VR Jan 2021

    """
    sizedat = np.shape(dcalc)
    nd = sizedat[0]
    rscal = Wd * (dobs - dcalc).T
    # print(sizedat,nd)
    # normalized root mean square error
    nrms = np.sqrt(np.sum(np.power(abs(rscal), 2)) / (nd - 1))
    # sum squared scaled symmetric error
    serr = 2.0 * nd * np.abs(rscal) / (abs(dobs.T) + abs(dcalc.T))
    ssq = np.sum(np.power(serr, 2))
    # print(ssq)
    srms = 100.0 * np.sqrt(ssq / nd)

    return nrms, srms

def nearly_equal(a,b,sig_fig=6):
    return (a==b or int(a*10**sig_fig) == int(b*10**sig_fig))

# def make_pdf_catalog(WorkDir="./", PdfList= None, FileName=None):
#     """
#     Make pdf catalog from site-plot(

#     Parameters
#     ----------
#     Workdir : string
#         Working directory.
#     Filename : string
#         Filename. Files to be appended must begin with this string.

#     Returns
#     -------
#     None.

#     """
#     # error("not in 3.9! Exit")

#     import fitz

#     catalog = fitz.open()

#     for pdf in PdfList:
#         with fitz.open(pdf) as mfile:
#             catalog.insert_pdf(mfile)

#     catalog.save(FileName, garbage=4, clean = True, deflate=True)
#     catalog.close()

#     print("\n"+str(np.size(PdfList))+" files collected to "+FileName)

def print_title(version="", fname="", form="%m/%d/%Y, %H:%M:%S", package="JacoPyAn",out=True):
    """
    Print version, calling filename, and modification date.
    """

    import os.path
    from datetime import datetime

    title = ""

    if len(version)==0:
        print("No version string given! Not printed to title.")
        tstr = ""
    else:
       ndat = "\n"+"".join("Date " + datetime.now().strftime(form))
       tstr =  package+" Version "+version+ndat+ "\n"

    if len(fname)==0:
        print("No calling filenane given! Not printed to title.")
        fstr = ""
    else:
        fnam = os.path.basename(fname)
        mdat = datetime.fromtimestamp((os.path.getmtime(fname))).strftime(form)
        fstr = fnam+", modified "+mdat+"\n"
        fstr = fstr + fname

    title = tstr+ fstr

    if out:
        print(title)

    return title

def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n


def KLD(P=np.array([]), Q=np.array([]), epsilon= 1.e-8):
    """
    Calculates Kullback-Leibler distance

    Parameters
    ----------
    P, Q: np.array
        pdfs
    epsilon : TYPE
        Epsilon is used here to avoid conditional code for
        checking that neither P nor Q is equal to 0.

    Returns
    -------

    distance: float
        KL distance


    """
    if P.size * Q.size==0:
        error("KLD: P or Q not defined! Exit.")

    # You may want to instead make copies to avoid changing the np arrays.
    PP = P.copy()+epsilon
    QQ = Q.copy()+epsilon

    distance = np.sum(PP*np.log(PP/QQ))

    return distance

def set_opacity(mesh=None, zones=[(-1., 1.), (-2., 0.8), (-3, 0.6), (-4., 0.2), (-5.,0.1 )]):

    dims = mesh.shape
    zons = zones.size

    opac = np.zeros_like(mesh)

    for izon in np.arange(zons):
        which = np.where(mesh>zones[izon][0])
        opac[which] = zones[izon][1]

    return opac

def make_pdf_catalog(PDFList= None, FileName="Catalog.pdf"):
    """
    Make pdf catalog from list of pdf plots

    Parameters
    ----------
    PDFList : List of strings
        List of Filenames
    Filename : string
        Catalog file


    Returns
    -------
    None.

    """
    # error("not in 3.9! Exit")

    import fitz

    catalog = fitz.open()

    for pdf in PDFList:
        with fitz.open(pdf) as mfile:
            catalog.insert_pdf(mfile)

    catalog.save(FileName, garbage=4, clean = True, deflate=True)
    catalog.close()

    print("\n"+str(np.size(PDFList))+" files collected to "+FileName)

def calc_rhoa_phas(freq=None, Z=None):

    mu0 = 4.0e-7 * np.pi  # Magnetic Permeability (H/m)
    omega = 2.*np*freq

    rhoa = np.power(np.abs(Z), 2) / (mu0 * omega)
    # phi = np.rad2deg(np.arctan(Z.imag / Z.real))
    phi = np.angle(Z, deg=True)

    return rhoa, phi

def mt1dfwd(freq, sig, d, inmod="r", out="imp", magfield="b"):
    """
    Calulate 1D magnetotelluric forward response.

    based on A. Pethik's script at www.digitalearthlab.com
    Last change vr Nov 20, 2020
    """
    mu0 = 4.0e-7 * np.pi  # Magnetic Permeability (H/m)

    sig = np.array(sig)
    freq = np.array(freq)
    d = np.array(d)

    if inmod[0] == "c":
        sig = np.array(sig)
    elif inmod[0] == "r":
        sig = 1.0 / np.array(sig)

    if sig.ndim > 1:
        error("IP not yet implemented")

    n = np.size(sig)

    Z = np.zeros_like(freq) + 1j * np.zeros_like(freq)
    w = np.zeros_like(freq)

    ifr = -1
    for f in freq:
        ifr = ifr + 1
        w[ifr] = 2.0 * np.pi * f
        imp = np.array(range(n)) + np.array(range(n)) * 1j

        # compute basement impedance
        imp[n - 1] = np.sqrt(1j * w[ifr] * mu0 / sig[n - 1])

        for layer in range(n - 2, -1, -1):
            sl = sig[layer]
            dl = d[layer]
            # 3. Compute apparent rho from top layer impedance
            # Step 2. Iterate from bottom layer to top(not the basement)
            #   Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt(1j * w[ifr] * mu0 * sl)
            wj = dj / sl
            #   Step 2.2 Calculate Exponential factor from intrinsic impedance
            ej = np.exp(-2 * dl * dj)

            #   Step 2.3 Calculate reflection coeficient using current layer
            #          intrinsic impedance and the below layer impedance
            impb = imp[layer + 1]
            rj = (wj - impb) / (wj + impb)
            re = rj * ej
            Zj = wj * ((1 - re) / (1 + re))
            imp[layer] = Zj

        Z[ifr] = imp[0]
        # print(Z[ifr])

    if out.lower() == "imp":

        if magfield.lower() =="b":
            return Z/mu0
        else:
            return Z

    elif out.lower() == "rho":
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))

        return rhoa, phase
    else:
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))
        return Z, rhoa, phase
