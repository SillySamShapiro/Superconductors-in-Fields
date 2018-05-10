#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:07:50 2018

@author: Sam J Shapiro
"""
#This script computes a special case of eqation (9) from Dr. Sasha Dukan's "Superconductivity in a high magnetic
#feild: Excitation spectrum and tunneling properties". We are working with the diagonal approximation (n=m), and setting
# j = 0. In the quadratic lattice b_x = 0 and b_y = a, which allows for simplification when we use a*b_y = a^2 = \pi*(l_H)^2.
#Equation (9) is defined in quasi-momentum space, so q_x and q_y are the coordinates, with -pi/2a < q_x < pi/2a
#and -pi/a < q_y < pi/a because of the lattice spacing. For convinience we will work with a mesh grid where -1 < x,y < 1,
#and let q_x = x*pi/2a, q_y = y*pi/a (note that all inequalities are non-strict). We have also used an identity that relates Hermite 
#poynomials of degree 2n to Laguerre polynomials of degree n. So in this case equation (9) is 
#$$\Delta^0_{nn}(q) = \Delta_{0}/\sqrt{2}\sum_{k} exp(2ik*\pi*y - (x/2 + k)^{2}\pi)L_{n}^{-1/2}(2(x/2 + k)^{2}\pi)$$,
#where L_{n}^{-1/2} is the nth Laguerre polynomial raised to the -1/2 power.

def gap_function_diagonal_quadratic(n, mesh_size , k_range , amplitude = 1):
    
    import numpy as np
    import scipy.special as sps
    
    mesh = np.meshgrid(np.linspace(-1,1,mesh_size),np.linspace(-1,1,mesh_size)) 
    output = np.zeros(mesh[0].shape)
    
    for k in range(-k_range, k_range):
        output = output + np.exp(np.pi*(2*1j*k*mesh[1] - (0.5*mesh[0] + k)**2))*sps.eval_genlaguerre(n, -0.5, 2*np.pi*((0.5*mesh[0] + k)**2))
        
    output = amplitude*output
    
    return output, mesh

out48, mesh2 = gap_function_diagonal_quadratic(48, 40, 10)

