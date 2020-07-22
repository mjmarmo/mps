# Copyright (c) 2019 Michael Joseph Marmo III. All rights reserved.
############################################################################################################################################################
# Project: 	 	MAGNETIC POSITION SYSTEM - 37                                                                                                              #
# Author:    	Michael Joseph Marmo III                                                                                                                   #
# Date:      	06-13-2019                                                                                                                                 #
# Version:   	4.2.1																																	   #
# File Name: 	mps_map.py 																																   #
# Objective: 	Global positioning system based on geomagnetic observations																				   #
# Description: 	Geospatial Mapping/Plotting                                                                                                                #
############################################################################################################################################################
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip as zip, count
#===========================================================================================================================================================
MPS_DATA_PATH = '/home/joe/Desktop/Project_MPS37/IGRF_Data/'
MPS_DATA_FILE = 'fl_data.txt'
#===========================================================================================================================================================

def geo_map(lon, lat):
		#plt.hold(True)
		plt.plot(-82.40286, 27.50134, 'bs')
		plt.plot(lon, lat, 'rs')
		plt.show()





#===========================================================================================================================================================