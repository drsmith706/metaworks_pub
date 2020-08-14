# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:24:11 2020

@author: drsmi
"""

import scipy as sc
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import linalg

class DataAnalysis:

    # Define a set of useful constants
    C     = scipy.constants.c
    EPS_0 = scipy.constants.epsilon_0 #C^2/(N*m^2)
    MU_0  = scipy.constants.mu_0    #m kg s^-2 A^-2
    cm    = 0.01
    GHz   = 1.0E9
    rad   = sc.pi/180.
    
    #Font style for plot titles:
    fontTitleStyle = {'family': 'sanserif',
                      'color': 'darkred',
                      'weight': 'normal',
                      'size': 16,
                      }
    
    def __init__(self, **kwargs):
    #Palettes used for plotting:
        palBlueTones = self.palettes()
        self.palBlueTones = palBlueTones
    
    def palettes(self):
        color_steps = 16
        r1, r2 = 0x10, 0xA5
        g1, g2 = 0x36, 0xB6
        b1, b2 = 0x9D, 0xE5
        
        r = np.linspace(r1, r2, color_steps)/255
        g = np.linspace(g1, g2, color_steps)/255
        b = np.linspace(b1, b2, color_steps)/255
        return np.transpose(np.vstack((np.vstack((r, g)), b)))
#   blueTones goes from 0x10369D to 0x099CE5
    
    def plot_1D(self, data, **kwargs):
        # Keyword arguments:
        # title (a string variable containing the plot title)
        # xlabel (a string variable describing the x-axis)
        # vs_angle (a boolean variable if the plot is as a function of angle)
        # vs_frequency (a boolean variable if the plot is as a function of frequency)
        # plot_size (sets plot size)
        # aspect_ratio (sets aspect ratio)
        
        xvals = np.real(data[0,:])
        rdat = data[1,:]
        if kwargs.get('color') != None:
            colorCurve = kwargs.get('color')
        elif kwargs.get('palette') != None:
            colorCurve = self.palBlueTones[kwargs.get('palette')]
        else:
            colorCurve = np.array((10,54,157))/255
        if kwargs.get('normalize') == True:
            rdat = rdat/np.amax(rdat)
        if kwargs.get('vs_angle') == True:
            xvals = xvals/self.rad
        elif kwargs.get('vs_frequency') == True:
            xvals = xvals/self.GHz            
        plotSize = kwargs.get('plot_size')
        if plotSize == None:
            plotSize = 8
        dpi=kwargs.get('dpi')
        if dpi == None:
            dpi = 100            
        r = kwargs.get('aspect_ratio')
        if r == None:
            r = 1.5
        # Generate the plot:
        if kwargs.get('suppress_plot') == True:
            ax = kwargs.get('plot_reference')
        else:
            fig, ax = plt.subplots()
            fig.set_dpi(dpi)
            fig.set_size_inches(plotSize,plotSize/r)
        fig = plt.gcf()
        scl = dpi * plotSize
#        size = fig.get_size_inches()*fig.dpi        
#        print(size)
        plt.sca(ax)
        plt.xticks(fontsize=np.int(0.014*scl))
        plt.yticks(fontsize=np.int(0.014*scl))        
        stitle = kwargs.get('title')
        if stitle != "":
            ax.set_title(stitle, fontsize=np.int(0.04*scl), pad=np.int(.04*scl))
        xlabel = kwargs.get('xlabel')
        if xlabel != "":
            ax.set_xlabel(xlabel, fontsize=np.int(0.02*scl),labelpad=15)
        ylabel = kwargs.get('ylabel')
        if ylabel != "":
            ax.set_ylabel(ylabel, fontsize=np.int(0.02*scl))
        else:
            ax.set_ylabel(fontsize=20)
            plt.yticks(fontsize=20)
        ax.plot(xvals, rdat, color=colorCurve)
    
    def plot_complex_polarizability_values(self, alph, **kwargs):
        if kwargs.get('normalize') == True:
            u = np.abs(alph)
            umax = np.max(u)
            alph = alph/umax
        plotSize = kwargs.get('plot_size')
        if plotSize == None:
            plotSize = 8
        ms = kwargs.get('marker_size')
        if ms == None:
            ms = 5
        dpi=kwargs.get('dpi')
        if dpi == None:
            dpi = 100  
        if kwargs.get('suppress_plot') == True:
            ax = kwargs.get('plot_reference')
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots() 
            fig.set_dpi(dpi)
            fig.set_size_inches(plotSize, plotSize)
        scl = dpi * plotSize
        plt.sca(ax)
        ax.axhline(y=0, color='k', lw=.1)
        ax.axvline(x=0, color='k',lw=.1)
        ax.set_aspect(1)
        stitle = kwargs.get('title')
        if stitle != "":
            ax.set_title(stitle, fontsize=np.int(0.024*scl), pad=np.int(.03*scl))          
        ax.scatter(np.real(alph),np.imag(alph),ms)
        plt.xticks(fontsize=np.int(0.014*scl))
        plt.yticks(fontsize=np.int(0.014*scl))
        ax.tick_params(length=4, width=.5)
#        ax.spines['left'].set_linewidth(1)
#        ax.spines['right'].set_linewidth(1)
#        ax.spines['top'].set_linewidth(1)
#        ax.spines['bottom'].set_linewidth(1)
        ax.set_xlabel(r'Re{$\alpha_m$}', fontsize=np.int(0.02*scl))
        ax.set_ylabel(r'Im{$\alpha_m$}', fontsize=np.int(0.02*scl))
        #plt.show
        
    def plot_density_2D(self,x1,x2,data, **kwargs):
        fig, ax = plt.subplots()
        plotSize = kwargs.get('plot_size')
        if plotSize == None:
            plotSize = 8
        dpi=kwargs.get('dpi')
        if dpi == None:
            dpi = 100
        scl = dpi * plotSize
        if kwargs.get('normalize') == True:
            rdat = np.abs(data)
            rdat = rdat/np.amax(rdat)
        stitle = kwargs.get('title')
        if stitle != None:
            ax.set_title(stitle,fontsize=np.int(0.04*scl), pad=np.int(.04*scl))
        fig.set_size_inches(plotSize, plotSize)
        fig.set_dpi(dpi)
        ax.set_xlabel(r'$\theta_x$ (degrees)', fontsize=20)
        ax.set_ylabel(r'$\theta_y$ (degrees)', fontsize=20)
        ax.tick_params(axis='x', labelsize=np.int(.02*scl), pad=np.int(.004*scl)) 
        ax.tick_params(axis='y', labelsize=np.int(.02*scl), pad=np.int(.004*scl)) 
        ctf = ax.pcolormesh(x1, x2, rdat)
        plt.show()

    def plot_alpha(self, SysArch, **kwargs):
        # Keyword arguments:
        # magnitude (true if magnitude is to be plotted)
        fig, ax = plt.subplots()
        if kwargs.get('magnitude')==True:
            rdat = np.abs(SysArch.alpha)
            ax.set_title('alpha, magnitude')
        elif kwargs.get('phase')==True:
            rdat=np.arctan(np.imag(SysArch.alpha)/np.real(SysArch.alpha))
            ax.set_title('alpha, phase')
        ax.set_xlabel('distance in cm')
        
        ax.plot(SysArch.positions_x,rdat)
        
    def plot_polar_1D(self, data, **kwargs):
        angles = np.add(np.real(data[0,:]),90)
        # Normalize the maximum point to 1:
        rdat = np.abs(data[1,:])
        rdat = rdat/np.amax(rdat)
        # Get the max and min of the log scale:
        scaleMin = kwargs.get('log_min')
        scaleMax = kwargs.get('log_max')
        rticks = kwargs.get('rticks')
        stitle = kwargs.get('title')
        plot_reference = kwargs.get('plot_reference')
        plotSizeX = kwargs.get('plot_size_x')
        plotSizeY = kwargs.get('plot_size_y')
        if plotSizeX == None:
            plotSizeX = 16
        if plotSizeY == None:
            plotSizeY = 18
        if kwargs.get('suppress_plot')==True:
            suppressPlot = True
        else:
            suppressPlot = False
        if scaleMin == None:
            scaleMin = -30
        if scaleMax == None:
            scaleMax = 0
        if rticks == None:
            rticks = 5
        # Convert to linear scale, then clip limits:
        scaleMinLin = 10.0**(scaleMin/10.0)
        rdat = np.clip(rdat,scaleMinLin,1)
        rdat = 10.*np.log10(rdat)
        if suppressPlot==True:
            ax = plot_reference
        else:
            ax = plt.subplot(111, projection='polar')
        ax.plot(angles*np.pi/180, rdat)
        ax.grid(True)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rticks(np.linspace(scaleMin, scaleMax, rticks))
        if stitle != "":
            ax2 = plt.gca()
            plt.text(0.5, .85, stitle, transform=ax2.transAxes, fontdict=self.fontTitleStyle,
                 horizontalalignment='center')
        fig = plt.gcf()
        fig.set_size_inches(plotSizeX, plotSizeY)
        if suppressPlot != True:
            plt.show()

    def plot_polar_2D_angle(self, phi, theta, data, **kwargs):
        #Normalize data to be directive gain
        u = self.calculate_directivity(theta, phi, data, calculation_type='energy')
        data = data * 4 * np.pi/u
        theta = theta/self.rad
        plotSize = kwargs.get('plot_size')
        dpi=kwargs.get('dpi')
        if dpi == None:
            dpi = 300
        if plotSize == None:
            plotSize = 8
        cmap = kwargs.get('color_map')
        if cmap == None:
            colorMap = 'jet'
        colorBar = kwargs.get('color_bar')
        if colorBar == None:
            colorBar = True    
        stitle = kwargs.get('title')
        if kwargs.get('suppress_plot')==True:
            suppressPlot = True
        else:
            suppressPlot = False
        plot_reference = kwargs.get('plot_reference')
        if suppressPlot==True:
            ax = plot_reference
        else:
            fig = plt.figure(dpi = dpi)
            ax = plt.subplot(111, polar=True)
        scl = dpi * plotSize
        ax.tick_params(axis='x', labelsize=np.int(.0024*scl), pad=np.int(.0002*scl))
        cmap = plt.get_cmap(colorMap)
        fig = plt.gcf()
        fig.set_size_inches(plotSize, plotSize)
        if stitle != "":
            ax.set_title(stitle, fontsize=np.int(0.004*scl), pad=np.int(.008*scl))
        ax.tick_params(axis='y', labelsize=np.int(0.0017*scl),  colors='w', rotation=0)
        ctf = ax.contourf(phi, theta, data, cmap=cmap)
        if colorBar == True:
            plt.colorbar(ctf, pad=.2, fraction = .03)
            ctf.colorbar.ax.tick_params(labelsize=np.int(.0017*scl)) 
        ax.set_rticks([15,30,45,60,75,90])
 #       ax.grid(True)
        if suppressPlot != True:
            plt.show()
        
    def plot_dipole_layout(self, sa, **kwargs):
        showDipoles = True
        showFeedField = False
        showPhase = False
        asSampled = False
        showArrayFactor = False
        showAlpha = False
        showColorBar = False
        
        if kwargs.get('show_phase') == True:
            showPhase = True
        else:
            showPhase = False
        if kwargs.get('show_magnitude') == True:
            showMagnitude = True
        else:
            showMagnitude = False
        if kwargs.get('show_dipoles') == True:
            showDipoles = True
        else:
            showDipoles = False
        if kwargs.get('show_feed_field') == True:
            showFeedField = True
        else:
            showFeedField = False

        if kwargs.get('show_array_factor') == True:
            if sa.alpha[0] == None:
                print('ERROR: No modulation pattern defined')
                showArrayFactor = False
                showFeedField = True
            else:
                showArrayFactor = True
                showFeedField = False
        else:
            showArrayFactor = False
        if kwargs.get('show_alpha') == True:
            if sa.alpha[0] == None:
                print('ERROR: No modulation pattern defined')
                showAlpha = False
                showFeedField = True
            else:
                showAlpha = True
                showArrayFactor = False
                showFeedPhase = False
                showFeedField = False
        else:
            showAlpha = False
        if kwargs.get('as_sampled') == True:
            asSampled = True
        else:
            asSampled = False
        if kwargs.get('show_color_bar') == True:
            showColorBar = True
        else:
            showColorBar = False
        pos_x = sa.positions_x/self.cm
        pos_y = sa.positions_y/self.cm
        ndipolesX = sa.numDipolesX
        ndipolesY = sa.numDipolesY
        spcx = sa.dipoleSpacingX/self.cm
        lx = spcx*(ndipolesX-1)+2*spcx
        dl = sa.wavelength_op/(self.cm * 20 * sa.guide_index) #Sample step size for quasi-continuous sampling

        if sa.numDipolesY > 1:
            ndipolesY = sa.numDipolesY
            spcy = sa.dipoleSpacingY/self.cm
            ly = spcy*ndipolesY+2*spcy
            wdth = (spcy/2)*.6
            pos_y = pos_y - (sa.numDipolesY-1)*spcy/2   #Center the dipoles about the y=0 axis for display
        else:
            spcy = spcx
            pos_y[0]=0
            ly = 3*spcx
            wdth = (spcy/2)*.6

#       Create the figure:                    
        plt.figure(figsize=(20,10),dpi=80)
        fscaled = 10*80 #The default scaling for all other figure elements
        plt.axes()
        ax = plt.gca()

#       Plot the layout as a rectangular region. The region extends beyond the actual active space by a little bit.
        rect = plt.Rectangle((-spcx,-ly/2),lx,ly,fc=(.722,.451,.2),zorder=1)
        ax.add_patch(rect)
        
        # Show dipoles as arrows
        # Note that the output of SystemArchitecture positions is arrays, and we just need to cycle over positions here.
        # So, extract single arrays of positions from the SystemArchitecture positions.
        if showDipoles == True:
            xfd = pos_x
            xfd.shape = (ndipolesY,ndipolesX)
            xfd = xfd[0,:]
            yfd = pos_y
            yfd.shape = (ndipolesX,ndipolesY)
            yfd = yfd[:,0]
            # Arrow formatting here:
            style="Simple, head_length=" + str(8) + ", head_width=" + str(9) + ", tail_width=" + str(1.5)
            for py in yfd:
                for px in xfd:
                    arrow = mpatches.FancyArrowPatch((px,py-wdth),(px+.001,py+wdth),
                                                     arrowstyle=style, color=(1,1,0), lw='2', zorder=10)
                    ax.add_patch(arrow)

        plt.axis('scaled')
        ax.set_xlabel('x-axis (cm)', fontsize=20)
        ax.set_ylabel('y-axis (cm)', fontsize=20)
        plt.ylim(-ly/2-spcy,ly/2+spcy)
        plt.xlim(-2*spcx,lx)
        plt.rcParams["axes.axisbelow"] = False

        
        # Plot the feed fields, sample in steps of 1/20 the guide wavelength
        if showFeedField == True:
            ti = 'Feed Wave Magnetic Field, '
            if showPhase == False and showMagnitude == False:
                ti = ti + 'Real Part'
                xfd = np.linspace(-spcx, lx-spcx, np.int(lx/dl))
                yfd = np.linspace(-ly/2, ly-ly/2, np.int(ly/dl))
                xf, yf = np.meshgrid(xfd, yfd)
                z=xf 
                plt.title(ti, fontsize=20, pad = 20)
                if sa.apertureDimension == 1:
                    z=np.real(sa.feed_architecture(xpos=xf*self.cm,sample=True))
                elif sa.apertureDimension == 2:
                    z=np.real(sa.feed_architecture_2D(xpos=xf*self.cm, ypos=yf*self.cm, sample=True))
                ctf = ax.pcolormesh(xf,yf,z/self.cm, cmap='bwr', zorder=2)
                if showColorBar == True:
                    plt.colorbar(ctf, pad=.05, fraction = .017)            
                    ctf.colorbar.ax.tick_params(labelsize=np.int(.03*80*6))
            # Plot the feed wave angle, sample in steps of 1/20 the guide wavelength, or show the actual phases sampled by the aperture
            else:
                if asSampled == False:
                    xfd = np.linspace(-spcx, lx-spcx, np.int(lx/dl))
                    yfd = np.linspace(-ly/2+spcy/2, ly-ly/2+spcy/2, 2)
                    xf, yf = np.meshgrid(xfd, yfd)
                else:
                    if sa.apertureDimension == 1:
                        xfd = (sa.positions_x)/self.cm - spcx/2
                        yfd = np.linspace(0, spcy, 2)
                        xf, yf = np.meshgrid(xfd, yfd)
                        newcol = np.transpose(xf[:,-1]+spcx)
                        xf = np.column_stack((xf,newcol))
                        newcol = yf[:,-1]
                        yf = np.column_stack((yf,newcol))     
                    elif sa.apertureDimension == 2:
                        xfd = pos_x - spcx/2
                        yfd = pos_y
                        xfd.shape = (ndipolesY,ndipolesX)
                        newcol = np.transpose(xfd[:,-1]+spcx)
                        newmat = np.column_stack((xfd,newcol))
                        newrow = newmat[-1,:]
                        xf = np.vstack((newmat,newrow))                
                        yfd.shape = (ndipolesY,ndipolesX)
                        newrow = yfd[-1,:]+spcy               
                        newmat = np.vstack((yfd, newrow))
                        newcol = newmat[:,-1]
                        yf = np.column_stack((newmat,newcol))
                        
                if sa.apertureDimension == 1:
                    if showPhase == True:
                        z=np.angle(sa.feed_architecture(xpos=xf*self.cm,sample=True))
                        ti = ti + 'Phase'
                    elif showMagnitude == True:
                        z=np.abs(sa.feed_architecture(xpos=xf*self.cm,sample=True))
                        ti = ti + 'Magnitude'
                    else:
                        z=np.real(sa.feed_architecture(xpos=xf*self.cm,sample=True))
                        ti = ti + 'Real Part'
                elif sa.apertureDimension == 2:
                    if showPhase == True:
                        z=np.angle(sa.feed_architecture(xpos=xf*self.cm, ypos=yf*self.cm, sample=True))
                        ti = ti + 'Phase'
                    elif showMagnitude == True:
                        z=np.abs(sa.feed_architecture(xpos=xf*self.cm, ypos=yf*self.cm, sample=True))
                        ti = ti + 'Magnitude'
                    else:
                        z=np.real(sa.feed_architecture(xpos=xf*self.cm, ypos=yf*self.cm, sample=True))
                        ti = ti + 'Real Part'
                plt.title(ti, fontsize=20, pad = 20)
                if asSampled == False:
                    ctf = ax.pcolormesh(xf,np.add(yf,-spcy/2),z/self.rad, cmap='Greys', zorder=2)
                else:
                    ctf = ax.pcolormesh(xf,np.add(yf,-spcy/2),z/self.rad, cmap='Greys', edgecolors='grey', linewidths=1, zorder=2)
                if showColorBar == True:
                    plt.colorbar(ctf, pad=.05, fraction = .017)            
                    ctf.colorbar.ax.tick_params(labelsize=np.int(.03*80*6))

        # Plot the array factor, which is the feed field multiplied by the polarizability at each sampled location
        if (showArrayFactor == True or showAlpha == True):
            if sa.apertureDimension == 1:
                xfd = (sa.positions_x)/self.cm - spcx/2
                yfd = np.linspace(0, spcy, 2)
                xf, yf = np.meshgrid(xfd, yfd)
                newcol = np.transpose(xf[:,-1]+spcx)
                xf = np.column_stack((xf,newcol))
                newcol = yf[:,-1]
                yf = np.column_stack((yf,newcol))

            elif sa.apertureDimension == 2:
                xfd = pos_x - spcx/2
                yfd = pos_y
                xfd.shape = (ndipolesY,ndipolesX)
                newcol = np.transpose(xfd[:,-1]+spcx)
                newmat = np.column_stack((xfd,newcol))
                newrow = newmat[-1,:]
                xf = np.vstack((newmat,newrow))                
                yfd.shape = (ndipolesY,ndipolesX)
                newrow = yfd[-1,:]+spcy               
                newmat = np.vstack((yfd, newrow))
                newcol = newmat[:,-1]
                yf = np.column_stack((newmat,newcol))
                
#                addedY = np.array(np.repeat(yfd[-1]+spcy,np.int(len(pos_x)/ndipolesY)))
#                yfd = np.hstack((pos_y, addedY))                
            if showArrayFactor == True:
                z = np.multiply(sa.hy, sa.alpha)
                ti = 'Array Factor'
            elif showAlpha == True:
                z = sa.alpha
                ti = 'Polarizability (alpha)'
            
            if showPhase == True:
                z = np.angle(z)
                cmapdef = 'Greys'
                clabel = "phase"
                ti = ti + ' , Phase'
            elif showMagnitude == True:
                z = np.abs(z)
                cmapdef = 'Greys'
                clabel = "magnitude"
                ti = ti + ' , Magnitude'
            else:
                z = np.real(z)
                cmapdef = 'bwr'
                clabel = "real part"
                ti = ti + ' , Real Part'
            
            if sa.apertureDimension == 1:
                z=np.tile(z,(len(yfd),1))
            else:
                z.shape = (ndipolesY,ndipolesX)
                newcol = z[:,-1]
                newmat = np.column_stack((z, newcol))
                newrow = newmat[-1,:]
                z = np.vstack((newmat,newrow))
                
            ctf = ax.pcolormesh(xf,np.add(yf,-spcy/2),z/self.rad, cmap=cmapdef, edgecolors='grey', linewidths=1,zorder=2)
            plt.title(ti, fontsize=20, pad = 20)
            if showColorBar == True:
                plt.colorbar(ctf, pad=.05, fraction = .017)            
                ctf.colorbar.ax.tick_params(labelsize=np.int(.03*80*6))
                ctf.colorbar.ax.get_yaxis().labelpad = 28
                ctf.colorbar.set_label(clabel, rotation=270, fontsize=18)



        # Draw the axes and style the grid and other plot elements       
        ax.set_facecolor((.9,.9,.9))
        ax.grid()
        ax.minorticks_on()
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=18)
          
        plt.show()

    def visualize_aperture(self, sa, **kwargs):
        #This doesn't work. MATPLOTLIB doesn't handle multiple 3D graphics without severe problems.
        spcX = sa.dipoleSpacingX/self.cm
        apertureSizeX = sa.apertureSizeX/self.cm
        apertureSizeY = 4*sa.dipoleSpacingX/self.cm
        
        p0=np.array([0,0,0])
        p1=np.array([apertureSizeX,apertureSizeY,-.5])
        
        fig = plt.figure(figsize=plt.figaspect(1)*4)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0,apertureSizeX/8)
        ax.set_ylim(-apertureSizeX/16,apertureSizeX/16)
        ax.set_zlim(0,apertureSizeX/8)
        
        #Draw the feed structure:
        self.drawCuboid(p0, p1, ax, alpha=1, color=(.3,.3,.3))
        
        #Draw the dipole positions:
        x = np.arange(spcX, apertureSizeX+spcX, spcX)
        y = np.zeros(len(x))+apertureSizeY/2
        z = np.ones(len(x))*.3
        
        u=0
        v=1
        w=0
        
        for i in range(len(x)):
            ax.plot([x[i],x[i]],[y[i]-.5,y[i]+.5],[z[i],z[i]],color='y',zorder=1,lw=.05)
                
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()        
        
    @staticmethod
    def calculate_beamwidth(data, dB_level: float = -3):
        angles = np.add(np.real(data[0,:]),90)
        # Normalize the maximum point to 1:
        rdat = np.abs(data[1,:])
        rdat = 10*np.log10(rdat/np.amax(rdat))
    
        indx_beam_max = np.argmax(rdat)

        indxs_left  = np.sort(np.argsort(np.absolute(rdat[:indx_beam_max+1]-dB_level))[:2])
        indxs_right = np.sort(np.argsort(np.absolute(rdat[indx_beam_max:]-dB_level))[:2])+1+indx_beam_max

        f=np.zeros([2,2])
        for row, indxs in enumerate([indxs_left,indxs_right]): 

            p = np.array([[angles[indxs[0]]],[rdat[indxs[0]]]]) 
            u = np.array([[angles[indxs[1]]],[rdat[indxs[1]]]]) 
                                      
            q = np.array([[0],[dB_level]])             
            v = np.array([[1],[dB_level]])
    
            B = p-q
    
            A = np.array([[v[0,0]-q[0,0], p[0,0]-u[0,0]],
                          [v[1,0]-q[1,0], p[1,0]-u[1,0]]])
    
            s = sc.linalg.solve(A,B) #x=inv(A)*B
        
            f[:,row:row+1] = p + s[1,0]*(u-p) 

        f1 = np.amax([f[0,0],np.amin(angles)]) 
        f2 = np.amin([f[0,1],np.amax(angles)]) 
        beamwidth = np.absolute(f2-f1)
        beam_direction = angles[indx_beam_max]
        
        #print(f'Beam direction: {beam_direction: .2f}')
        #print(f'Calculated {dB_level: 2.1f} dB beamwidth: {beamwidth: .2f}')
        return (beamwidth, beam_direction, f1, f2)
    
    @staticmethod
    def calculate_directivity(theta, phi, rp, **kwargs):
        calculationType = kwargs.get('calculation_type')
        thetaM, phiM = np.meshgrid(theta, phi)        
        u = 0
        i=1
        while i<theta.size:
            j=1
            while j<phi.size:
                dphi = phiM[i,j]-phiM[i-1,j]
                dtheta = (thetaM[i,j]-thetaM[i,j-1])*np.sin((thetaM[i,j]+thetaM[i,j-1])/2)
                u = u + rp[j,i]*dphi*dtheta
                j=j+1
            i=i+1
        rp_max = np.max(rp)
        directivity = 4*np.pi*rp_max/u
        if calculationType == 'energy':
            return u
        elif calculationType == 'directivity':
            return directivity
        elif calculationType == 'directivitydb':
            return 10*np.log10(directivity)
    
    @staticmethod
    def draw_cuboid(p0, p1, ax, **kwargs):
        
        if kwargs.get('alpha') !=None:
            alpha = kwargs.get('alpha')
        else:
            alpha = 1
        if kwargs.get('color') != None:
            color = kwargs.get('color')
        else:
            color = 'b'
        
        x0m=np.full((2,2),p0[0])
        x1m=np.full((2,2),p1[0])
        y0m=np.full((2,2),p0[1])
        y1m=np.full((2,2),p1[1])
        z0m=np.full((2,2),p0[2])
        z1m=np.full((2,2),p1[2])

        x0101, x0011=np.meshgrid([p0[0],p1[0]],[p0[0],p1[0]])
        y0101, y0011=np.meshgrid([p0[1],p1[1]],[p0[1],p1[1]])
        z0101, z0011=np.meshgrid([p0[2],p1[2]],[p0[2],p1[2]])

        ax.plot_surface(x0011, y0m, z0101, alpha=alpha, color=color)
        ax.plot_surface(x0011, y1m, z0101, alpha=alpha, color=color)
        ax.plot_surface(x0011, y0101, z1m, alpha=alpha, color=color)
        ax.plot_surface(x0011, y0101, z0m, alpha=alpha, color=color)
        ax.plot_surface(x0m, y0011, z0101, alpha=alpha, color=color)
        ax.plot_surface(x1m, y0011, z0101, alpha=alpha, color=color)        
