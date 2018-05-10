#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:07:50 2018

@author: Sam J Shapiro
"""

#%%
#This script computes a special case of eqation (9) from Dr. Sasha Dukan's "Superconductivity in a high magnetic
#feild: Excitation spectrum and tunneling properties". We are working with the diagonal approximation (n=m), and setting
# j = 0. In the quadratic lattice b_x = 0 and b_y = a, which allows for simplification when we use a*b_y = a^2 = \pi*(l_H)^2.
#Equation (9) is defined in quasi-momentum space, so q_x and q_y are the coordinates, with -pi/2a < q_x < pi/2a
#and -pi/a < q_y < pi/a because of the lattice spacing. For convinience we will work with a mesh grid where -1 < x,y < 1,
#and let q_x = x*pi/2a, q_y = y*pi/a (note that all inequalities are non-strict). We have also used an identity that relates Hermite 
#poynomials of degree 2n to Laguerre polynomials of degree n. So in this case equation (9) is 
#$$\Delta^0_{nn}(q) = \Delta_{0}/\sqrt{2}\sum_{k} exp(2ik*\pi*y - (x/2 + k)^{2}\pi)L_{n}^{-1/2}(2(x/2 + k)^{2}\pi)$$,
#where L_{n}^{-1/2} is the nth Laguerre polynomial raised to the -1/2 power.

def gap_function_diagonal_quadratic(n, k_range, mesh_size, amplitude = 1):
    
    import numpy as np
    import scipy.special as sps
    
    mesh = np.meshgrid(np.linspace(-1,1,mesh_size),np.linspace(-1,1,mesh_size)) 
    output = np.zeros(mesh[0].shape)
    
    for k in range(-k_range, k_range):
        output = output + np.exp(np.pi*(2*1j*k*mesh[1] - (0.5*mesh[0] + k)**2))*sps.eval_genlaguerre(n, -0.5, 2*np.pi*((0.5*mesh[0] + k)**2))
        
    output = np.abs(output)
    
    output = amplitude*output
    
    return output, mesh

gap0, mesh0 = gap_function_diagonal_quadratic(0, 15, 200, amplitude = 1)

#%%
# The following function is basically the same as the first, but for a triangular lattice
def gap_function_diagonal_triangular(n, k_range, mesh_size, amplitude = 1):
    
    import numpy as np
    import scipy.special as sps
    
    mesh = np.meshgrid(np.linspace(0-1,1,mesh_size),np.linspace(-1,1,mesh_size))
    output = np.zeros(mesh[0].shape)
    
    for k in range(-k_range, k_range):
        output = output + np.exp(0.5*1j*np.pi*(k**2) + 1j*np.pi*k*mesh[1] - 0.5*np.sqrt(3)*np.pi*(0.5*mesh[0] + k)**2)*sps.eval_genlaguerre(n, -0.5, np.sqrt(3)*np.pi*(mesh[0] + k)**2)
    
    output = np.abs(output)
    
    output = amplitude*output
    
    return output, mesh

gap0t, mesh0t = gap_function_diagonal_triangular(0, 15, 200, amplitude = 1)
#%%
def gap_function(n, m, k_range, mesh_size, lattice = 'quadratic', amplitude = 1):
    
    import numpy as np
    import scipy.special as sps
    
    s = m - n
    
    if lattice == 'quadratic':
        mesh = np.meshgrid(np.linspace(-1,1,mesh_size),np.linspace(-1,1,mesh_size)) 
        output = np.zeros(mesh[0].shape)
        
        if s == 0: #the case when n == m
            for k in range(-k_range, k_range):
                output = output + np.exp(np.pi*(2*1j*k*mesh[1] - (0.5*mesh[0] + k)**2))*sps.eval_genlaguerre(n, -0.5, 2*np.pi*((0.5*mesh[0] + k)**2))
        
            output = np.abs(output)
            output = amplitude*output
            return output, mesh
        
        elif s%2 == 0 and s > 0: #the case when m = n + 2N for some natural number N
            
    
        
    else:   
        output = 0
    return