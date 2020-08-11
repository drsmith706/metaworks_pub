# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:13:47 2020

@author: drsmi
"""

import scipy as sc
import numpy as np
import scipy.constants
from scipy import linalg

class ElementProperties:
    # The ElementProperties class defines the relevant properties of an individual metamaterial element.
    
    # Define a set of useful constants
    C     = scipy.constants.c
    EPS_0 = scipy.constants.epsilon_0 #C^2/(N*m^2)
    MU_0  = scipy.constants.mu_0    #m kg s^-2 A^-2
    cm    = 0.01
    GHz   = 1.0E9
    
    # If no input parameters provided, use a default set.
    # For the default, assume the operating wavelength is 10 GHz.
    # The antenna is 1D, lambda/4 spacing of 31 elements (0.75 cm).
    
    
    def __init__(self, **kwargs):

        self._dipoleType   = 'ideal-unconstrained'
        self._a = 1
        self._fop = None
        self._f0 = 10*self.GHz
        self._g = 0.2*self.GHz
        self._q = self._f0/self._g
        self._tuningFreqLow = self._f0 - 3*self._g/2
        self._tuningFreqHigh = self._f0 + 3*self._g/2
        self.minAngle = None
        self.maxAngle = None
        

    def getDipoleType(self):
        return self._dipoleType
    
    def getPolarizability(self, f, **kwargs):
        if kwargs.get('scan_resonance') != True:
            alpha = self._a * f**2/(self._f0**2 - f**2 + 1j*f*self._g)
        else:
            if kwargs.get('operating_frequency') != None:
                fop = kwargs.get('operating_frequency')
                alpha = self._a * fop**2/(f**2 - fop**2 + 1j*fop*self._g)
#                print(fop/self.GHz)
#                print(f/self.GHz)
#                print(alpha)
            else:
                print('ERROR: operating_frequency must be set if scan_resonance is used.')
        return alpha

#   Define or get the properties of an ideal Lorentzian dipole resonator.        
    def setLorentzianDipoleParameters(self, **kwargs):
        a = kwargs.get('oscillator_strength')
        if a != None:
            self._a = a
        f0 = kwargs.get('resonance_frequency')
        if f0 != None:
            self._f0 = f0
        if kwargs.get('Q') != None:
            self._q = kwargs.get('Q')
            self._g = self._f0/self._q
        else:
            g = kwargs.get('damping_frequency')
            if g != None:
                self._g = g
                self._q = self._f0/self._g
        if kwargs.get('operating_frequency') != None:
            self._fop = kwargs.get('operating_frequency')                
        if kwargs.get('tuning_frequency_low') != None:
            self._tuningFreqLow = kwargs.get('tuning_frequency_low')
        if kwargs.get('tuning_frequency_high') != None:
            self._tuningFreqHigh = kwargs.get('tuning_frequency_high')
        if self._fop != None:
            alpha = self.getPolarizability(self._tuningFreqLow, scan_resonance=True, operating_frequency=self._fop)
            self.minAngle = np.angle(alpha)
            self.AlphaMin = alpha
            alpha = self.getPolarizability(self._tuningFreqHigh, scan_resonance=True, operating_frequency=self._fop)
            self.maxAngle = np.angle(alpha)
            self.AlphaMax = alpha
            

        

    def getLorentzianDipoleParameters(self, **kwargs):
        return self._f0, self._g, self._a, self._q

    def summarizeLorentzianDipoleParameters(self):
        print('Lorentzian Dipole Parameters:')
        print('Resonance Frequency: ' + str(self._f0/self.GHz) + ' GHz')
        print('Damping Frequency: ' + str(self._g/self.GHz) + ' GHz')
        print('Oscillator Strength: ' + str(self._a))
        print('Quality Factor (Q): ' + str(self._q))
        print('Tuning frequency, minimum: ' + str(self._tuningFreqLow/self.GHz) + ' GHz')
        print('Tuning frequency, maximum: ' + str(self._tuningFreqHigh/self.GHz) + ' GHz')