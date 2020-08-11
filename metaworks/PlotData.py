# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:24:11 2020

@author: drsmi
"""

import scipy as sc
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
from scipy import linalg

class PlotData:

    # Define a set of useful constants
    C     = scipy.constants.c
    EPS_0 = scipy.constants.epsilon_0 #C^2/(N*m^2)
    MU_0  = scipy.constants.mu_0    #m kg s^-2 A^-2
    cm    = 0.01
    GHz   = 1.0E9
    rad   = sc.pi/180.
    
    def Plot2D(self, data):
        fig, ax=plt.subplots()
        ax.plot(np.real(data[0,:])/self.rad,np.abs(data[1,:]))