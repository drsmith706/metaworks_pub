# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:56:10 2020

@author: drsmith
"""

import scipy as sc
import numpy as np
import scipy.constants

class SystemOutput:
    # The SystemArchitecture class defines an arbitrary dipole layout. 
    # Excitation field is found from the FeedArchitecture class.
    
    # Define a set of useful constants
    C     = scipy.constants.c
    EPS_0 = scipy.constants.epsilon_0 #C^2/(N*m^2)
    MU_0  = scipy.constants.mu_0    #m kg s^-2 A^-2
    cm    = 0.01
    GHz   = 1.0E9
    rad   = sc.pi/180.
    
    def array_factor(self, SysArch, **kwargs):
        # The ArrayFactor attribute has several input parameters.
        # angle_start is the start angle in degrees
        # angle_stop is the stop angle in degrees
        # angle_num is the number of angles
        
        phi_start  = kwargs.get('angle_start')*self.rad
        phi_end    = kwargs.get('angle_stop')*self.rad
        phi_num    = kwargs.get('angle_num')

        angle = np.linspace(phi_start,phi_end,phi_num)
        af_vs_angle = np.ones(len(angle),dtype=complex)
        
        i=0
#        for angle_n in angle:
#            argm = -(SysArch.betaX + SysArch.k0*np.sin(angle_n))*SysArch.positions_x        
#            afTerms = np.multiply(SysArch.alpha, np.exp(np.multiply(1j,argm)))
#            af_vs_angle[i] = np.sum(afTerms)
 #           i=i+1

        for angle_n in angle:
            w = SysArch.alpha*SysArch.hy
            argm = -(SysArch.k0*np.sin(angle_n))*SysArch.positions_x        
            afTerms = np.multiply(w, np.exp(np.multiply(1j,argm)))
            af_vs_angle[i] = np.sum(afTerms)
            i=i+1
        
        return np.vstack((angle, af_vs_angle))

    def radiation_pattern(self, SysArch, **kwargs):
        # The RadiationPattern method has several input parameters.
        # theta_start is the start angle in degrees
        # theta_stop is the stop angle in degrees
        # theta_num is the number of angles
        
        theta_start  = kwargs.get('theta_start')*rad
        theta_end    = kwargs.get('theta_stop')*rad
        theta_num    = kwargs.get('theta_num')

        angle = np.linspace(theta_start,theta_end,theta_num)
        rp_vs_angle = np.ones(len(angle))
        
        for i, angle_n in enumerate(angle):
            w = SysArch.alpha*SysArch.hy
            argm = -(SysArch.k0*np.sin(angle_n))*SysArch.positions_x        
            afTerms = np.multiply(w, np.exp(np.multiply(1j,argm)))
            rp = np.abs(np.sum(afTerms))
            rp_vs_angle[i] = rp**2

        
        angle = angle*180/np.pi
        return np.vstack((angle, rp_vs_angle))

    def radiation_pattern_2D(self, SysArch, **kwargs):
        # The ArrayFactor attribute has several input parameters.
        # angle_start is the start angle in degrees
        # angle_stop is the stop angle in degrees
        # angle_num is the number of angles
        
        theta_start  = kwargs.get('theta_start')*self.rad
        theta_end    = kwargs.get('theta_stop')*self.rad
        theta_num    = kwargs.get('theta_num')
        
        scanType = kwargs.get('scan_type')
        
        if scanType == None:
            scanType = 'xy'
            print('No scan type has been selected; defaulting to kx-ky scan type.')
        elif scanType == 'theta_slice':
            angleTheta = np.linspace(theta_start, theta_end,theta_num)
            anglePhi = kwargs.get('phi_0')
            if anglePhi==None:
                anglePhi=0
                print('No  phi angle selected for theta scan; phi=0 default used.')
            anglePhi = anglePhi * self.rad
            rp_vs_angle = np.ones(len(angleTheta),dtype=complex)
        
            for i, angle_n in enumerate(angleTheta):
                w=np.multiply(SysArch.alpha, SysArch.hy)
                argm = np.add(-(SysArch.k0*np.sin(angle_n)*np.cos(anglePhi))*SysArch.positions_x, -(SysArch.k0*np.sin(angle_n)*np.sin(anglePhi))*SysArch.positions_y)
                afTerms = np.multiply(w,np.exp(np.multiply(1j,argm)))
                af = np.sum(afTerms)
                rp_vs_angle[i] = np.abs(af)**2
        
            return np.vstack((angleTheta*180/np.pi, rp_vs_angle))
        elif scanType == 'angle':
            phi_start  = kwargs.get('phi_start')*self.rad
            phi_end    = kwargs.get('phi_stop')*self.rad
            phi_num    = kwargs.get('phi_num')
            angleTheta = np.linspace(theta_start,theta_end,theta_num)
            anglePhi = np.linspace(phi_start,phi_end,phi_num)
            rp_vs_angle = np.ones((len(angleTheta),len(anglePhi)),dtype=complex)
        
            w = np.multiply(SysArch.alpha, SysArch.hy)
            m=0
            for angle_m in anglePhi:
                n=0
                for angle_n in angleTheta:
                    argm = np.add(-(SysArch.k0*np.sin(angle_n)*np.cos(angle_m))*SysArch.positions_x, -(SysArch.k0*np.sin(angle_n)*np.sin(angle_m))*SysArch.positions_y)        
                    afTerms = np.multiply(w, np.exp(np.multiply(1j,argm)))
                    af = np.sum(afTerms)
                    rp_vs_angle[n,m] = np.abs(af)**2
                    n=n+1
                m=m+1
            return angleTheta, anglePhi, np.real(rp_vs_angle)
        elif scanType == 'xy':
            phi_start  = kwargs.get('phi_start')*self.rad
            phi_end    = kwargs.get('phi_stop')*self.rad
            phi_num    = kwargs.get('phi_num')

            angleTheta = np.linspace(theta_start,theta_end,theta_num)
            anglePhi = np.linspace(phi_start,phi_end,phi_num)
            x = np.cos(angleTheta)
            y = np.cos(anglePhi)            
#            x = np.linspace(-.9,.9,200)
#            y = np.linspace(-.9,.9,200)
            rp_vs_xy = np.ones((len(x),len(y)),dtype=complex)
        
            w = np.multiply(SysArch.alpha, SysArch.hy)
            m=0
            for x_m in x:
                n=0
                for y_n in y:
                    argm = np.add(-SysArch.k0*x_m*SysArch.positions_x, -SysArch.k0*y_n*SysArch.positions_y)        
                    afTerms = np.multiply(w, np.exp(np.multiply(1j,argm)))
                    af = np.sum(afTerms)
                    rp_vs_xy[n,m] = np.abs(af)**2
                    n=n+1
                m=m+1
            return angleTheta/self.rad, anglePhi/self.rad, np.real(rp_vs_xy)
