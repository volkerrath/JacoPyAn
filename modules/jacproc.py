#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:36:08 2020

"""
from sys import exit as error
import numpy as np
import scipy.sparse as scs
import numpy.linalg as npl



def calc_sensitivity(Jac=np.array([]),
                     Type = "euclidean", UseSigma = False, OutInfo = False):
    """
    Calculate sensitivities.
    Expects that Jacobian is already sclaed, i.e Jac = C^(-1/2)*J.

    Several options exist for calculating sensiotivities, all of them
    used in the literature.
    Type:
        "raw"     sensitivities summed along the data axis
        "abs"     absolute sensitivities summed along the data axis
                    (often called coverage)
        "euc"     squared sensitivities summed along the data axis.
        "cum"     cummulated sensitivities as proposed by
                  Christiansen & Auken, 2012. Not usable for negative data.

    Usesigma:
        if true, sensitivities with respect to sigma  are calculated.

    Christiansen, A. V. & Auken, E.
    A global measure for depth of investigation
    Geophysics, 2012, 77, WB171-WB177

    from UBC:
    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(Utils.sdiag(1/delta_d/pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S-thres_hold > 0.
        doi = abs(self.survey.depth[active]).max()
        return doi, active

    T. Guenther
    Inversion Methods and Resolution Analysis for the 2D/3D Reconstruction
    of Resistivity Structures from DC Measurements
    Fakultaet für Geowissenschaften, Geotechnik und Bergbau,
    Technische Universitaet Bergakademie Freiberg, 2004.

    author:VR 9/23

    """

    if np.size(Jac)==0:
        error("calc_sensitivity: Jacobian size is 0! Exit.")

    if UseSigma:
        Jac = -Jac



    if "raw" in  Type.lower():
        S = Jac.sum(axis=0)
        if OutInfo:
            print("raw:", S)
        # else:
        #     print("raw sensitivities")
        # smax = Jac.max(axis = 0)
        # smin = Jac.max(axis = 0)
        
    elif "cov" in Type.lower():
        S = Jac.abs().sum(axis=0)
        if OutInfo:
            print("cov:", S)
        # else:
        #     print("coverage")

    elif "euc" in Type.lower():
        S = Jac.power(2).sum(axis=0)
        if OutInfo:
            print("euc:", S)
        # else:
        #     print("euclidean (default)")

    elif "cum" in Type.lower():
        S = Jac.abs().sum(axis=0)
        # print(np.shape(S))
        # S = np.sum(Jac,axis=0)

        S = np.append(0.+1.e-10, np.cumsum(S[-1:0:-1]))
        S = np.flipud(S)
        if OutInfo:
           print("cumulative:", S)
        # else:
        #    print("cumulative sensitivity")

    else:
        print("calc_sensitivity: Type "
              +Type.lower()+" not implemented! Default assumed.")
        S = Jac.power(2).sum(axis=0)

        if OutInfo:
            print("euc (default):", S)
        # else:
        #     print("euclidean (default)")

        # S = S.reshape[-1,1]
 
    S=S.A1 
    return S


def transform_sensitivity(S=np.array([]), V=np.array([]),
                          Transform=["size","max", "sqrt"],
                          asinhpar=[0.], OutInfo=False):
    """
    Transform sensitivities.

    Several options exist for transforming sensitivities, all of them
    used in the literature.

    Normalize options:
        "siz"       Normalize by the values optional array V ("volume"), 
                    i.e in our case layer thickness. This should always 
                    be the first value in Transform list.
        "max"       Normalize by maximum value.
        "sur"       Normalize by surface value.
        "sqr"       Take the square root. Only usefull for euc sensitivities. 
        "log"       Take the logaritm. This should always be the 
                    last value in Transform list
        "asinh"     asinh transform. WARNING: excludes all other options, and should be used
                    only for raw sensitivities
        

    author:VR 4/23

    """

    if np.size(S)==0:
        error("transform_sensitivity: Sensitivity size is 0! Exit.")
    
    ns = np.shape(S)
    

    for item in Transform:       
        
        if "siz" in item.lower():
             print("trans_sensitivity: Transformed by layer thickness.")
             if np.size(V)==0:
                 error("Transform_sensitivity: No thicknesses given! Exit.")

             else:
                 V = V.reshape(ns)
                 S = S/V
                 # print("S0v", np.shape(S))
                 # print("S0v", np.shape(V))
                 
        if "max" in item.lower():
             print("trans_sensitivity: Transformed by maximum value.")
             maxval = np.amax(np.abs(S))
             print("maximum value: ", maxval)
             S = S/maxval
             # print("S0m", np.shape(S))
            
        if "sqr" in item.lower():
            S = np.sqrt(S)
            # print("S0s", np.shape(S))
            
        if "log" in item.lower():    
            S = np.log10(S)

        if "asinh" in item.lower():
            maxval = np.amax(S)
            minval = np.amin(S)
            if maxval>0 and minval>0:
                print("transform_sensitivity: No negatives, switched to log transform!")
                S = np.log10(S)
            else:
                if len(asinhpar)==1:
                    scale = asinhpar[0]
                else:
                    scale = get_scale(S, method=asinhpar[0])

                    S = np.arcsinh(S/scale)

    
    # S = S.A1    
    # print(type(S))
    
    return S

def get_scale(d=np.array([]), F=0.1, method = "other", OutInfo = False):
    """
    Get optimal Scale for arcsin transformation.

    Parameters
    ----------
    d : float, required.
        Data vector.
    F : float, optional
        Weight for arcsinh transformation, default from Scholl & Edwards (2007)

    Returns
    -------
    S : float
        Scale value for arcsinh

    C. Scholl
    Die Periodizitaet von Sendesignalen bei Long-Offset Transient Electromagnetics
    Diploma Thesis, Institut für Geophysik und Meteorologie der Universität zu Koeln, 2001.


    """

    if np.size(d)==0:
        error("get_S: No data given! Exit.")

    if "s2007" in method.lower():
        scale = F * np.nanmax(np.abs(d))

    else:
        dmax = np.nanmax(np.abs(d))
        dmin = np.nanmin(np.abs(d))
        denom =F *(np.log(dmax)-np.log(dmin))
        scale = np.abs(dmax/denom)

    if OutInfo:
        print("Scale value S is "+str(scale)+", method "+method)

    return scale

def sparsmat_to_arry(mat=None):
    """
    

    Parameters
    ----------
    mat : sparse scipy matrix
        sparse scipy matrix. The default is None.
    arr : np.array
         The default is np.array([]).

    Returns
    -------
    arr : Tnp.array
         The default is np.array([]).YPE


    """
    arr = np.array([])
    
    data = mat.A1
    
    return arr


def update_avg(k = None, m_k=None, m_a=None, m_v=None):
    """
    Update the mean and variance from data stream.

    Note: final variance needs to be divided by k-1.

    Based on the formulae from
    Knuth, Art of Computer Programming, Vol 2, page 232, 1997.

    VR  Mar 7, 2021

    Note: generalization possible for skewness (M3) and kurtosis (M4)

    delta = x - M1;
    delta_n = delta / n;
    delta_n2 = delta_n * delta_n;
    term1 = delta * delta_n * n1;
    M1 += delta_n;
    M4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
    M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
    M2 += term1;

    """
    if k == 1:
        m_avg = m_k
        m_var = np.zeros_like(m_avg)

    md = m_k - m_a
    m_avg = m_a + md/np.abs(k)
    m_var = m_v + md*(m_k - m_avg)

    if k < 0:
        m_var = m_var/(np.abs(k-1))

    return m_avg, m_var

# def update_med(k = None, model_n=None, model_a=None, model_v=None):
#     """
#     Estimate the quantiles from data stream.

#     T-digest

#     VR  Mar , 2021
#     """

#     return m_med, m_q1, m_q2

def rsvd(A, rank=300, n_oversamples=None, n_subspace_iters=None, return_range=False):
    """
    =============================================================================
    Randomized SVD. See Halko, Martinsson, Tropp's 2011 SIAM paper:

    "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions"
    Author: Gregory Gundersen, Princeton, Jan 2019
    =============================================================================
    Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
    """
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A.
    # print(' stage A')
    Q = find_range(A, n_samples, n_subspace_iters)

    # Stage B.
    # print(' stage B')
    B = Q.T @ A
    # print(np.shape(B))
    # print(' stage B before linalg')
    U_tilde, S, Vt = np.linalg.svd(B)
    # print(' stage B after linalg')
    U = Q @ U_tilde

    # Truncate.
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    # This is useful for computing the actual error of our approximation.
    if return_range:
        return U, S, Vt, Q
    return U, S, Vt


# ------------------------------------------------------------------------------


def find_range(A, n_samples, n_subspace_iters=None):
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    """
    # print('here we are in range-finder')
    m, n = A.shape
    O = np.random.randn(n, n_samples)
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)


# ------------------------------------------------------------------------------


def subspace_iter(A, Y0, n_iters):
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    # print('herere we are in subspace-iter')
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T @ Q)
        Q = ortho_basis(A @ Z)
    return Q


# ------------------------------------------------------------------------------


def ortho_basis(M):
    """Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    """
    # print('herere we are in ortho')
    Q, _ = np.linalg.qr(M)
    return Q


def sparsify_jac(Jac=None, 
                 sparse_thresh=1.0e-6, normalized=False, scalval = 1., 
                 method=None, out=True):
    """
    Sparsifies error_scaled Jacobian from ModEM output

    author: vrath
    last changed: Sep 10, 2023
    """
    shj = np.shape(Jac)
    if out:
        nel = shj[0] * shj[1]
        print(
            "sparsify_jac: dimension of original J is %i x %i = %i elements"
            % (shj[0], shj[1], nel)
        )
        
    Jf = Jac.copy()
    
    if scalval <0.:
        Scaleval = np.amax(np.abs(Jf))
        print("sparsify_jac: output J is scaled by %g (max Jacobian)"
            % (Scaleval))  
    else: 
        Scaleval = abs(scalval)
        print("sparsify_jac: output J is scaled by %g" % (Scaleval))  
    
    Jf[np.abs(Jf)/Scaleval < sparse_thresh] = 0.0

    Js = scs.csr_matrix(Jf)
    #Js = scs.lil_matrix(Jf)

    if out:
        ns = Js.count_nonzero()
        print("sparsify_jac:"
                +" output J is sparse, and has %i nonzeros, %f percent"
                % (ns, 100.0 * ns / nel))
        test = np.random.normal(size=np.shape(Jac)[1])
        normx = npl.norm(Jf@test)
        normo = npl.norm(Jf@test-Js@test)
        print(normx, normo, normo/normx)
        normd = npl.norm((Jac-Jf), ord="fro")
        normf = npl.norm(Jac, ord="fro")
        # print(norma)
        # print(normf)
        print(" Sparsified J explains "
              +str(round(100.-100.*normd/normf,2))+"% of full J.")
        print("****", nel, ns, 100.0 * ns / nel, round(100.-100.*normd/normf,3) )

    if normalized:
        f = 1.0 / Scaleval
        Js = normalize_jac(Jac=Js, fn=f)
        
    #Js = Js.tocsr()

    return Js, Scaleval


def normalize_jac(Jac=None, fn=None, out=True):
    """
    normalize Jacobian from ModEM data err.

    author: vrath
    last changed: Sep30, 2023
    """
    shj = np.shape(Jac)
    shf = np.shape(fn)
    print("fn = ")
    print(fn)
    if shf[0] == 1:
        f = 1.0 / fn[0]
        Jac = f * Jac
    else:
        erri = scs.diags(1./fn[:], 0, format="csr")
        Jac = erri @ Jac
        #erri = np.reshape(1.0 / fn, (shj[0], 1))
        #Jac = erri[:] * Jac

    return Jac

def set_mask(rho=None, pad=[10, 10 , 10, 10, 0, 10], blank= np.nan, flat=True, out=True):
    """
    Set model masc for Jacobian calculations.

    author: vrath
    last changed: Dec 29, 2021

    """
    shr = np.shape(rho)
    # jm = np.full(shr, np.nan)
    jm = np.full(shr, blank)
    print(np.shape(jm))

    jm[pad[0]:-pad[1], pad[2]:-pad[3], pad[4]:-pad[5]] = 1.
    # print(pad[0], -1-pad[1])
    # jt =jm[0+pa-1-pad[1]-1-pad[1]d[0]:-1-pad[1], 0+pad[2]:-1-pad[3], 0+pad[4]:-1-pad[5]]
    # print(np.shape(jt))
    mask = jm
    if flat:
        # mask = jm.flatten()
        mask = jm.flatten(order="F")

    return mask



# def calculate_sens(Jac=None, normalize=False, small=1.0e-15, blank=np.nan, log=False, out=True):
#     """
#     Calculate sensitivity from ModEM Jacobian.
#     Optionally blank elements smaller than theshold, take logarithm, normalize.

#     author: vrath
#     last changed: Sep 25, 2020
#     """
#     if scs.issparse(Jac):
#         J = Jac.todense()
#     else:
#         J = Jac

#     S = np.sum(np.power(J, 2), axis=0)
#     S = np.sqrt(S)

#     Smax = np.nanmax(S)
#     Smin = np.nanmin(S)

#     if out:
#         print("Range of S is "+str(Smin)+" to "+str(Smax))

#     if normalize:
#         S = S / Smax
#         if out:
#             print("Normalizing with"+str(Smax))

#     if log:
#         S = np.log10(S)

#     S[S < small] = blank

#     return S, Smax


def project_model(m=None, U=None, small=1.0e-14, out=True):
    """
    Project to Nullspace.

    (see Munoz & Rath, 2006)
    author: vrath
    last changed: Sep 25, 2020
    """
    b = np.dot(U.T, m)
    # print(m.shape)
    # print(b.shape)
    # print(U.shape)

    mp = m - np.dot(U, b)

    return mp


def transfrom_model(m=None, M=None, small=1.0e-14, out=True):
    """
    Transform Model.

    M should be something like C_m^-1/2
    ( see eg Kelbert 2012, Egbert & kelbert 2014)
    author: vrath
    last changed:  Oct 12, 2020
    """
    transm = np.dot(M, m)

    return transm
