# Copyright (c) 2019 Michael Joseph Marmo III. All rights reserved.  
############################################################################################################################################################
# Project: 	 	MAGNETIC POSITION SYSTEM                                                                                                                   #
# Author:    	Michael Joseph Marmo III                                                                                                                   #
# Date:      	06-013-2019                                                                                                                                #
# Version:   	4.2.1																																	   #
# File Name: 	mps_tools.py 																															   #
# Objective: 	Global positioning system based on geomagnetic observations																				   #
# Description: 	Data Processing Tools                                                                                                                      #
############################################################################################################################################################
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip as zip, count
#===========================================================================================================================================================
import numpy as np

mps_data_path = '/home/joe/Desktop/Project_MPS37/IGRF_Data/'

mps_data_file = 'I_Grid_2015.mf'

'''
# USA Boundary Coordinates
lon_min = -119
lon_max = -68
lat_min = 24
lat_max = 45
'''
# Florida Boundary Coordinates
lon_min = -84
lon_max = -81
lat_min = 24
lat_max = 30

def extract_data(filename, lon_min, lon_max, lat_min, lat_max):
	
	raw_data = np.loadtxt(fname=filename)
	
	ext_lon = raw_data[np.where((raw_data[:,0] >= lon_min) * (raw_data[:,0] <= lon_max))]

	ext_lat = ext_lon[np.where((ext_lon[:,1] >= lat_min) * (ext_lon[:,1] <= lat_max))]

	ext_data = ext_lat

	print ext_data[:,0].min()
	print ext_data[:,0].max()
	print ext_data[:,1].min()
	print ext_data[:,1].max()

	print ext_data.shape

	return ext_data
