# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:52:08 2020

@author: drsmith

Class: SystemArchitecture

The SystemArchitecture class creates objects that contain all aspects of the system, including the dipole layout
and any feed structure.

"""

import scipy as sc
import numpy as np
import scipy.constants
from metaworks.ElementProperties import ElementProperties
from scipy import linalg

class SystemArchitecture:
    # The SystemArchitecture class defines an arbitrary dipole layout and feed structure. 
    # Note that to keep consistent with the usual definitions of phi and theta, the antenna aperture is in the xz-plane
    # for 2D apertures, with y being the broadside direction. This means that the phi angle is in the xy plane,
    # as usually defined, with phi=90 degrees being the broadside direction and phi=0 being the positive x-axis.
    
    # Define a set of useful constants
    C     = scipy.constants.c
    EPS_0 = scipy.constants.epsilon_0 #C^2/(N*m^2)
    MU_0  = scipy.constants.mu_0    #m kg s^-2 A^-2
    cm    = 0.01
    GHz   = 1.0E9
    
    # If no input parameters provided, use a default set.
    # For the default, assume the operating wavelength is 10 GHz.
    # The antenna is 1D, lambda/4 spacing of 31 elements (0.75 cm).
    
    freq_op_default = 10*GHz
    positions_x_default, dipoleSpacingX_default = np.linspace(0.0*cm, 22.5*cm, 31, retstep=True)
    positions_y_default = np.zeros(31)
    dipoleID_default    = np.arange(31)
    alpha_default       = np.ones(31)
    dipole_layout_default = np.vstack((dipoleID_default,positions_x_default,positions_y_default,alpha_default))
    
    # For the feed properties, what matters is the field at the position of each dipole.
    # The type of feed changes the calculation for the feed mode, with the field at each dipole finally saved.
    
    feed_type_default = 'plane wave'
    
    
    def __init__(self, freq_op: float = freq_op_default, feed_type: str = feed_type_default,
             dipole_layout: list = dipole_layout_default):

        self.freq_op      = freq_op
        self.wavelength_op=self.C/self.freq_op
        self.dipoleID     = dipole_layout[0,:]
        self.positions_x  = dipole_layout[1,:]
        self.positions_y  = dipole_layout[2,:] 
        self.alpha        = dipole_layout[3,:]
        self.dipoleLayout = dipole_layout
        self.numDipolesX   = len(self.dipoleID)
        self.numDipolesY   = 1
        self._dipoleType   = 'ideal-unconstrained'
        self.dipoleProperties = ElementProperties()
        self.layoutType   = 'linear 1D'
        self.apertureDimension = 1
        self.dipoleSpacingX= self.positions_x[1]-self.positions_x[0]
        self.dipoleSpacingY= None
        self.apertureSizeX= self.dipoleSpacingX * self.numDipolesX
        self.apertureSizeY= None
        self.modulationType = None
        
        #
        # Feed attributes
        self.feed_type    = feed_type
        self._guideIndex   = 2.5
        self.betaX        = None
        self.betaY        = None
        self.k0           = 2*sc.pi*self.freq_op/self.C
        self.nx           = 1.0
        self.ny           = 0.0
        self.hy           = np.ones(self.numDipolesX)
        
        # Calculate the incident field distribution for the default feed structure:
        self.FeedArchitecture(feed_type = self.feed_type, set=True)

#   The following set of functions allow access to class properties, allowing validation of input and self-constinency when parameters are changed.        

    @property
    def dipoleType(self):
        return self._dipoleType

    @dipoleType.setter
    def dipoleType(self, value):
        selection = None
        properties = ('ideal', 'ideal-unconstrained', 'ideal-magnitude-only', 'ideal-constrained-lorentzian',
                      'lorentzian-limited-tuning')
        if value == 'help':
            print('Supported dipole types are: ideal, ideal-unconstrained, ideal-magnitude-only, ideal-constrained-lorentzian')
            return
        else:
            for property in properties:
                if value == property:
                    selection = value
        if selection != None:
            self._dipoleType = selection
        else:
            print(value + ' is not a supported type')
        if value == 'lorentzian-limited-tuning':
            self.dipoleProperties.setLorentzianDipoleParameters(tuning_frequency_low = 9.5*self.GHz, tuning_frequency_high=10.5*self.GHz,
                                                                operating_frequency=self.freq_op)
    
    @property
    def guideIndex(self):
        return self._guideIndex

    @guideIndex.setter
    def guideIndex(self, value):
        self._guideIndex = value
        if self.apertureDimension == 1:
            self.FeedArchitecture(feed_type = self.feed_type, set=True)
        else:
            self.FeedArchitecture2D(feed_type=self.feed_type, set=True)

    def MakeLinearArray(self, spc, **kwargs):
        self.apertureDimension = 1
        apSize = kwargs.get('aperture_size')
        num = kwargs.get('number_elements')
        self.layoutType = 'linear 1D'
        if apSize == None:
            apSize = (num-1)*spc
        else:
            num = int(np.floor(apSize/spc)+1)
        self.dipoleSpacingX = spc
        self.apertureSizeX = apSize
        self.positions_x = np.linspace(0.0*self.cm, spc*(num-1), num)
        self.numDipolesX  = num
        self.numDipolesY = 1
        self.dipoleID    = np.arange(num)
        self.positions_y = np.zeros(num)
        self.alpha       = np.ones(num)
        self.dipoleLayout= np.vstack((self.dipoleID, self.positions_x, self.positions_y, self.alpha))

        self.FeedArchitecture(self.feed_type, set=True)

    def MakeLinear2DArray(self, spcx, spcy, **kwargs):
        self.apertureDimension=2
        self.layoutType = 'linear 2D'
        apSizeX = kwargs.get('aperture_size_x')
        numx = kwargs.get('number_elements_x')
        if apSizeX == None:
            apSizeX = (numx-1)*spcx
        else:
            numx = np.floor(apSizeX/spcx)+1
        apSizeY = kwargs.get('aperture_size_y')
        numy = kwargs.get('number_elements_y')
        if apSizeY == None:
            apSizeY = (numy-1)*spcy
        else:
            numy = np.floor(apSizeY/spcy)+1
        self.dipoleSpacingX = spcx
        self.dipoleSpacingY = spcy
        self.apertureSizeX = apSizeX
        self.apertureSizeY = apSizeY
        u, v = np.mgrid[0.0*self.cm:spcy*(numy-1):1j*numy,0.0*self.cm:spcx*(numx-1):1j*numx]
        self.positions_x = v.ravel()
        self.positions_y = u.ravel()
        self.numDipoles  = numx*numy
        self.numDipolesX = numx
        self.numDipolesY = numy
        self.dipoleID    = np.arange(self.numDipoles)
        self.alpha       = np.ones(self.numDipoles)
        self.dipoleLayout= np.vstack((self.dipoleID, self.positions_x, self.positions_y, self.alpha))
        
        self.FeedArchitecture2D(feed_type=self.feed_type, set=True)
   
        
    def SummarizeParameters(self):
        print('Operating Frequency: ' + str(self.freq_op/self.GHz) + ' GHz')
        print('Operating Wavelength: ' + str(self.wavelength_op/self.cm) + ' cm')
        print('Dipole Type: ' + self._dipoleType)
        print('Layout Type: ' + self.layoutType)
        if self.apertureDimension == 1:
            print('Dipole Spacing: ' + str(self.dipoleSpacingX/self.cm) + 'cm')
            print('Aperture Size: ' + str(self.apertureSizeX/self.cm) + 'cm')
            print('Number of Dipoles: ' + str(self.numDipolesX))
        else:
            print('Dipole Spacing along x: ' + str(self.dipoleSpacingX/self.cm) + 'cm')
            print('Dipole Spacing along y: ' + str(self.dipoleSpacingY/self.cm) + 'cm')
            print('Aperture Size along x: ' + str(self.apertureSizeX/self.cm) + 'cm')
            print('Aperture Size along y: ' + str(self.apertureSizeY/self.cm) + 'cm')
            print('Number of Dipoles along x: ' + str(self.numDipolesX))
            print('Number of Dipoles along y: ' + str(self.numDipolesY))            
        print('Feed Type: ' + self.feed_type)
        print('Waveguide Index: ' + str(self.guideIndex))
        print('Modulation Type: ' + str(self.modulationType))
    
    def FeedArchitecture(self, xpos=0, **kwargs):
        # Based on the feed type, select the appropriate function. The functions corresponding to different
        # feeds are specified below.
        
        def planeWave(xpos=0, **kwargs):
            if kwargs.get('set') == True:
                self.betaX = (2*sc.pi*self.freq_op/self.C)*self.guideIndex*self.nx
                self.hy = np.exp(-np.multiply(1j,self.betaX*self.positions_x))
            elif kwargs.get('sample') == True:
                return np.exp(-np.multiply(1j, np.multiply(self.betaX, xpos)))
        
        def microstrip():
            return self.freq * 1

        def rectWaveguide():
            return self.freq * .5

        choices = {
            'plane wave': planeWave,
            'microstrip': microstrip,
            'rectangular waveguide': rectWaveguide 
        }
        
        if kwargs.get('feed_type') != None:
            feed_type = kwargs.get('feed_type')
            feedFunc = choices.get(feed_type)
        else:
            feed_type = self.feed_type
            feedFunc = choices.get(feed_type)
        if kwargs.get('set') == True:
            feedFunc(set=True)
        elif kwargs.get('sample') == True:
            return feedFunc(xpos, sample=True)
                
    def FeedArchitecture2D(self, xpos=0, ypos=0, **kwargs):
        # Based on the feed type, select the appropriate function. The functions corresponding to different
        # feeds are specified below.
        
        def planeWave2D(xpos=0, ypos=0, **kwargs):
            if kwargs.get('set') == True:
                self.betaX = (2*sc.pi*self.freq_op/self.C)*self.guideIndex*self.nx
                self.betaY = (2*sc.pi*self.freq_op/self.C)*self.guideIndex*self.ny
                self.hy = np.multiply(np.exp(-np.multiply(1j,self.betaX*self.positions_x)), np.exp(-np.multiply(1j,self.betaY*self.positions_y)))
            elif kwargs.get('sample') == True:
                return np.multiply(np.exp(-np.multiply(1j,np.multiply(self.betaX,xpos))), np.exp(-np.multiply(1j,np.multiply(self.betaY, ypos))))
            
        def microstrip2D():
            self.freq = freq_op
            return self.freq * 1

        def rectWaveguide2D():
            self.freq = freq_op
            return self.freq * .5

        choices = {
            'plane wave': planeWave2D,
            'microstrip': microstrip2D,
            'rectangular waveguide': rectWaveguide2D 
        }
        
        if kwargs.get('feed_type') != None:
            feed_type = kwargs.get('feed_type')
            feedFunc = choices.get(feed_type)
        else:
            feed_type = self.feed_type
            feedFunc = choices.get(feed_type)
        if kwargs.get('set')==True:
            feedFunc(set=True)
        elif kwargs.get('sample') == True:
            return feedFunc(xpos, ypos, sample=True)   

    
