# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:06:02 2020

@author: drsmith
"""

import scipy as sc
import numpy as np
import scipy.constants
from scipy import linalg

class ModulationPattern:
    # The ModulationPattern class determines the polarizability distributions for desired beam patterns. 
    # Different assumptions about the polarizability constraints can be included.
    
    # Define a set of useful constants
    C     = scipy.constants.c
    EPS_0 = scipy.constants.epsilon_0 #C^2/(N*m^2)
    MU_0  = scipy.constants.mu_0    #m kg s^-2 A^-2
    cm    = 0.01
    GHz   = 1.0E9
    rad   = sc.pi/180.
    
    # The accepted keyword arguments and their values are as follows:
    # PatternType (directed beam)

        
    def __init__(self, SysArch,**kwargs):

        self.modulationType = kwargs.get('modulation_type')
        if self.modulationType != None:
            SysArch.modulationType = self.modulationType

    def directed_beam(self, s, angleSteer,**kwargs):
        angleSteer = angleSteer*self.rad
        arg = np.sin(angleSteer)*s.k0
        if self.modulationType == None:
            print('ModulationPattern must first be initialized with the modulation_type keyword')
            return

        # Internal function to compute the Euclidean map
        def _euclidean_map(alpha, **kwargs):
            r = kwargs.get('lorentzian_amplitude')
            if r == None:
                r = 0.5
            xi = np.real(alpha)
            yi = np.imag(alpha)
            if xi == 0:
                if yi >= 0:
                    x = 0
                    y = 0
                else:
                    x = 0
                    y = -2*r
            elif xi >= 0:
                if yi == -r:
                    x = r
                    y = -r
                else:
                    m = (yi+r)/xi
                    x = r*(1+m**2)**(-0.5)
                    if yi <= -r:
                        y = -r*(1+(1/m)**2)**(-0.5)-r
                    elif yi >= -r:
                        y = r*(1+(1/m)**2)**(-0.5)-r
            elif xi <= 0:
                if yi == -r:
                    x = -r
                    y = -r
                else:
                    m = (yi+r)/xi
                    x = -r*(1+m**2)**(-0.5)
                    if yi <= -r:
                        y = -r*(1+(1/m)**2)**(-0.5)-r
                    elif yi >= -r:
                        y = r*(1+(1/m)**2)**(-0.5)-r
            return (x+1j*y)
    # Main program loop starts here:

        if s.dipole_type == 'ideal-unconstrained':
            s.alpha = np.exp(np.multiply(1j,(s.betaX+arg)*s.positions_x))
        elif s.dipole_type == 'ideal-magnitude-only':
            xo = kwargs.get('mag_offset')
            if xo == None:
                xo = 1.0
            mo = kwargs.get('mag_modulation')
            if mo == None:
                mo = 0.5
            print(xo)
            print(mo)
            s.alpha = xo+mo*np.cos((s.betaX+arg)*s.positions_x)
        elif s.dipole_type == 'ideal-constrained-lorentzian':
            if self.modulationType == 'af-optimized':
                s.alpha = np.add(-1j, np.multiply(1,np.exp(np.multiply(1j,(s.betaX+arg)*s.positions_x))))/2.0
            elif self.modulationType == 'euclidean-optimized':
                s.alpha = np.exp(np.multiply(1j,(s.betaX+arg)*s.positions_x))
                for i in range(len(s.alpha)):
                    s.alpha[i] = _euclidean_map(s.alpha[i])
        elif s.dipole_type == 'lorentzian-limited-tuning':
            if self.modulationType == 'af-optimized':
                s.alpha = np.add(-1j, np.multiply(1,np.exp(np.multiply(1j,(s.betaX+arg)*s.positions_x))))/2.0                
                for i in range(len(s.alpha)):
                    u = s.dipoleProperties.minAngle
                    if np.angle(s.alpha[i]) <= u:
                        s.alpha[i] = -np.multiply(np.sin(u), np.exp(np.multiply(1j,u)))
                    u = s.dipoleProperties.maxAngle
                    if np.angle(s.alpha[i]) >= u:
                        s.alpha[i] = -np.multiply(np.sin(u), np.exp(np.multiply(1j,u)))


    def directed_beam_2D(self, s, anglePhi, angleTheta, **kwargs):
        anglePhi = anglePhi*self.rad
        angleTheta = angleTheta*self.rad
        argx = np.sin(angleTheta)*np.cos(anglePhi)*s.k0
        argy = np.sin(angleTheta)*np.sin(anglePhi)*s.k0
        if self.modulationType == None:
            print('ModulationPattern must first be initialized with the modulation_type keyword')
            return

        # Internal function to compute the Euclidean map
        def _euclidean_map_2D(alpha, **kwargs):
            r = kwargs.get('lorentzian_amplitude')
            if r == None:
                r = 0.5
            xi = np.real(alpha)
            yi = np.imag(alpha)
            if xi == 0:
                if yi >= 0:
                    x = 0
                    y = 0
                else:
                    x = 0
                    y = -2*r
            elif xi >= 0:
                if yi == -r:
                    x = r
                    y = -r
                else:
                    m = (yi+r)/xi
                    x = r*(1+m**2)**(-0.5)
                    if yi <= -r:
                        y = -r*(1+(1/m)**2)**(-0.5)-r
                    elif yi >= -r:
                        y = r*(1+(1/m)**2)**(-0.5)-r
            elif xi <= 0:
                if yi == -r:
                    x = -r
                    y = -r
                else:
                    m = (yi+r)/xi
                    x = -r*(1+m**2)**(-0.5)
                    if yi <= -r:
                        y = -r*(1+(1/m)**2)**(-0.5)-r
                    elif yi >= -r:
                        y = r*(1+(1/m)**2)**(-0.5)-r
            return (x+1j*y)
    # Main program loop starts here:
        if s.dipole_type == 'ideal-unconstrained':
            s.alpha = np.multiply(np.exp(np.multiply(1j,(s.betaX+argx)*s.positions_x)),np.exp(np.multiply(1j,(s.betaY+argy)*s.positions_y)))
        elif s.dipole_type == 'ideal-magnitude-only':
            xo = kwargs.get('mag_offset')
            if xo == None:
                xo = 1.0
            mo = kwargs.get('mag_modulation')
            if mo == None:
                mo = 0.5
            print(xo)
            print(mo)
            s.alpha = xo+mo*np.cos((s.betaX+arg)*s.positions_x)
        elif s.dipole_type == 'ideal-constrained-lorentzian':
            if self.modulationType == 'af-optimized':
                s.alpha = np.add(1j, np.multiply(1j,np.exp(np.multiply(1j,(s.betaX+arg)*s.positions_x))))/2.0
            elif self.modulationType == 'euclidean-optimized':
                s.alpha = np.exp(np.multiply(1j,(s.betaX+arg)*s.positions_x))
                for i in range(len(s.alpha)):
                    s.alpha[i] = _euclidean_map(s.alpha[i])
 

            

   # def ConstrainedLorentzianModulation(self, arg)