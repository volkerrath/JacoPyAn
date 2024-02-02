# JacoPyAn

_**NOTE: This repository is work in progress, not all scripts have been finished and tested. Use at your own risk! However, we will succesively 
update the repository correcting bugs, and adding additional functionality.**_

This repository contains the tools for manipulating, processing, analysing Jacobians from 3-D magnetotelluric inversion, including 
randomized singular value decompositions (SVD), and other tools related to the Nullspace-Shuttle method [1, 2]. Currently, We implemented the tools necessary
for using Jacobians generated by ModEM [3, 4]. Mainly the routines for manipulating/sparsifying the full Jacobian, the SVD, and the calculation of sensitivity 
can be used.

**Preprocessing the Jacobian**

The Jacobian  of a data and parameter set is defined as $J_{ij} = \dfrac{\delta d_i}{\delta m_j}$. Before being able to use it for further
action, a  few steps are necessary. ModEM seeks the MAP solution to the usual Bayesianinverse problem defined by:

```math
\[\begin{gathered}
  \Theta  = {({\mathbf{g}}({\mathbf{p}}) - {\mathbf{d}})^T}{\mathbf{C}}_{\mathbf{d}}^{ - 1}({\mathbf{g}}({\mathbf{p}}) - {\mathbf{d}}) + {({\mathbf{p}} - {{\mathbf{p}}_a})^T}{\mathbf{C}}_{\mathbf{p}}^{ - 1}({\mathbf{p}} - {{\mathbf{p}}_a}) \\ 
   = \left\| {{\mathbf{C}}_{\mathbf{d}}^{ - 1/2}({\mathbf{g}}({\mathbf{p}}) - {\mathbf{d}})} \right\|_2^2 + \left\| {{\mathbf{C}}_{\mathbf{p}}^{ - 1/2}({\mathbf{p}} - {{\mathbf{p}}_a})} \right\|_2^2 \\ 
\end{gathered} \]
```
However, it uses a transformation of data and parameter, given by:



**Sensitivities**

The use of sensitivities (in a variety of flavours) is comparatively easy, but needs some clarification, as it does not really conform 
to the everyday use of the word. Sensitivities are derived from the final model Jacobian matrix, which often is available from the inversion algorithm 
itself. It needs to be kept in mind that this implies any conclusions drawn are valid in the domain of validity for the Taylor expansion involved only. 
This may be a grave disadvantage in highly non-linear settings, but we believe that it still can be usefull for fast characterization of uncertainty.

Here, the parameter vector $\mathbf(m)$ is the natural logarithm of resistivity. This Jacobian is first normalized with the data error 
to obtain $\mathbf{\tilde{J}}$. While this procedure is uncontroversial, the definition of 
sensitivity is not unique, and various forms can be found in the literature, and $\texttt{JacoPyAn}$ calculates several of them:


1. "Raw" sensitivities, defined as $S_j = \sum_{i=1,n_d} \tilde{J}_{ij}$. No absolute values are involved, hence there may be 
both, positive and negative, elements. This does not conform to what we expect of sensitivity (positivity), but carries the most direct 
information on the role of parameter $j$ in the inversion.

2. "Euclidean" sensitivities, which are the most commonly used form. They are is defined as: 
$S^2_j = \sum_{i=1,n_d} \left|\tilde{J}_{ij}\right|^2=diag\left(\mathbf{\tilde{J}}^T\mathbf{\tilde{J}}\right)$.
This solves the positivity issue of raw sensitivities. The square root of this sensitivity is often preferred, and implemented in 
many popular inversion codes. 
    
3. Coverage. For this form, the absolute values of the Jacobian are used: $\sum_{i=1,n_d} \left|\tilde{J}_{ij}\right|$

For a definition of a depth of investigation (DoI), or model blanking/shading, forms (2) and (3) can be used. This, however, requires the 
choice of a threshold/scale is required, depending on the form applied. 

When moving from the error-normalised Jacobian,$\mathbf{J}_d$ to sensitivity, there are more choices for further normalisation, depending 
on the understanding and use of this parameter. If sensitivity is to be interpreted as an approximation to a continuous field over the 
volume of the model, it seems useful normalize by the cell volume. On the other hand, effect of the size is important when investigating 
the true role of this cell in the inversion. Finally, for comparing different data (sub)sets, it is convenient to do a final 
normalization by the maximum value in the model. All these options are implemented in the $\texttt{JacoPyAn}$ toolbox. 

_[1] M. Deal and G. Nolet (1996)_ 
_“Nullspace shuttles"_ 
_Geophysical Journal International, 124, 372–380_

_[2] G. Muñoz and V. Rath (2006)_
_“Beyond smooth inversion: the use of nullspace projection for the exploration of non-uniqueness in MT"_
_Geophysical Journal International, 164, 301–311, 2006, doi: 10.1111/j.1365-246X.2005.02825.x_

_[3] G. D. Egbert and A. Kelbert (2012)_ 
_“Computational recipes for electromagnetic inverse problems”_ 
_Geophysical Journal International, 189, 251–267, doi: http://dx.doi.org/10.1111/j.1365-246X.2011.05347.x_

_[4] A. Kelbert, N. Meqbel, G. D. Egbert, and K. Tandon (2014)_
_“ModEM: A Modular System for Inversion of Electromagnetic Geophysical Data”_
_Computers & Geosciences, 66, 440–53, doi: http://dx.doi.org/10.1016/j.cageo.2014.01.010_


  
 
