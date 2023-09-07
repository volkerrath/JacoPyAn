# JacoPyAn


**NOTE: This repository is work in progress, not all scripts have been finished and tested. Use at your own risk! However, we will succesively 
update the repository correcting bugs, and adding additional functionality.**

This repository contains the tools for manipulating, processing, analysing Jacobians from 3-D magnetotelluric inversion, including 
randomized singular value decompositions (SVD), and other tools related to the Nullspace-Shuttle method. Currently mainly the routines 
for manipulating/sparsifying the full Jacobian, and the calculation of sensitivity can be used.

The use of sensitivities (in a variety of flavours) is comparatively easy, but needs some clarification, as it does not really conform 
to the everyday use of the word. Sensitivities are derived from the final model Jacobian matrix, which often is available from the inversion algorithm 
itself. It needs to be kept in mind that this implies any conclusions drawn are valid in the domain of validity for the Taylor expansion involved only. 
This may be a grave disadvantage in highly non-linear settings, but we believe that it still can be usefull for fast characterization of uncertainty.

The Jacobian  of a data and parameter set is defined as

$$J_{ij} = \dfrac{\delta f_d(d_i)}{\delta f_m(m_j)} = \dfrac{\delta m_j}{\delta f_m(m_j)
    }\dfrac{\delta f_d(d_i)}{\delta d_i}
    \cdot \dfrac{\delta d_i}{\delta m_j} $$

Here, the parameter vector $\mathbf(m)$ is the natural logarithm of resistivity. This Jacobian is first normalized with the data error 
(i. e., multiplied by $\mathbf{C}^{-1/2}_d $) to obtain $\mathbf{\tilde{J}}$. While this procedure is uncontroversial, the definition of 
sensitivity is not unique, and various forms can be found in the literature, and \texttt{JacoPyAn} calculates several of them:


(1) "Raw" sensitivities, defined as:
    $$    
    S_j = \sum_{i=1}^{n_d} \tilde{J}_{ij}
    $$
    No absolute value is involved, hence there may be both, positive and negative, values. This does not conform to what we expect of 
    sensitivity (positivity), but carries the most direct information on the role of parameter $j$ in the inversion.

(2) "Euclidean" sensitivities, which are the most commonly used form. They are is defined as:
    $$
    \mathbf{S}^2_j = \sum_{i=1}^{n_d} \left|\tilde{J}_{ij}\right|^2=diag\left(\mathbf{\tilde{J}}^T\mathbf{\tilde{J}}\right)$$
    $$
    This solves the positivity issue of raw sensitivities. The square root of this sensitivity is often preferred, and implemented in 
    many popular inversion codes. 
(3) Coverage. For this form, the absolute values of the Jacobian are used:   
    $$
    \mathbf{S}_j = \sum_{i=1}^{n_d} \left|\tilde{J}_{ij}\right|
    $$

For a definition of a depth of investigation (DoI), or model blanking/shading, forms (2) and (3) can be used. This, however, requires the 
choice of a threshold/scale is required, depending on the form applied. 

When moving from the error-normalised Jacobian $\mathbf{J}_d$ to sensitivity, there are more choices for further normalisation, depending on the understanding and use of this parameter: 
If sensitivity is to be interpreted as an approximation to a continuous field over the volume of the model, it seems useful normalize by the cell volume. 
On the other hand, effect of the size is important when investigating the true role of this cell in the inversion. Finally, for comparing different data (sub)sets, it is convenient to do a final 
normalization by the maximum value in the model. All these options are implemented in the \texttt{JacoPyAn} toolbox. 
