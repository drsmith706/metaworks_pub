# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:26:46 2020

@author: drsmith
"""
from SystemArchitecture import SystemArchitecture
from ModulationPattern import ModulationPattern
from SystemOutput import SystemOutput
from DataAnalysis import DataAnalysis

x=SystemArchitecture()


modPatt=ModulationPattern(x)
modPatt.DirectedBeam(x, 30)


sysOut=SystemOutput()
af=sysOut.ArrayFactor(x,angle_start=-70,angle_stop=70,angle_num=50)

dt=DataAnalysis()
dt.Plot2D(af)