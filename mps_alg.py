# Copyright (c) 2019 Michael Joseph Marmo III. All rights reserved.
############################################################################################################################################################
# Project: 	 	MAGNETIC POSITION SYSTEM - 37                                                                                                              #
# Author:    	Michael Joseph Marmo                                                                                                                       #
# Date:      	06-09-2019                                                                                                                                 #
# Version:   	4.2.1																																	   #
# File Name: 	mps_alg.py 																																   #
# Objective: 	Global positioning system based on geomagnetic observations																				   #
# Description: 	RBF vector field/Ordinary Kriging Algorithms                                                                                               #
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
# Magnetometer Sensor (uT) - iPhone 4s/Sensor Kinetics 
Bx_0 = 15e-3 
By_0 = 30e-3 
Bz_0 = -51e-3
#===========================================================================================================================================================
# Radial Basis Function (RBF) Neural Network for Geospatial Interpolation of Divergence-free Vector Fields
class RBF(object):

	def __init__(self, MPS_DATA_FILE='/home/joe/Desktop/Project_MPS37/IGRF_Data/fl_data.txt', EPSILON=1000, BETA=0.5):
		self.network = np.loadtxt(MPS_DATA_FILE)
		self.epsilon = EPSILON
		self.beta = BETA
		self.N = self.network.shape[0]
	
	# Multiquadratic RBF Kernel w/ Euclidean Metric
	def basis(self, i, j):
		self.metric = (self.network[i,0+3]-self.network[j,0+3])**2 + (self.network[i,1+3]-self.network[j,1+3])**2 + (self.network[i,2+3]-self.network[j,2+3])**2
		return np.power((1 + self.epsilon*self.metric), self.beta)
	# 1st-Order Partial Derivatives   
	def basis_uv(self, i, j, u, v):
		metric = (self.network[i,0+3]-self.network[j,0+3])**2 + (self.network[i,1+3]-self.network[j,1+3])**2 + (self.network[i,2+3]-self.network[j,2+3])**2
		del_coeff = (self.network[i,u+3]-self.network[j,u+3]) * (self.network[i,v+3]-self.network[j,v+3])
		return 4 * (self.epsilon**4) * self.beta * (self.beta - 1) * del_coeff * np.power((1 + self.epsilon*metric), self.beta - 2)
	# 2nd-Order Partial Derivatives 
	def basis_uu(self, i, j, u, v):
		metric = (self.network[i,0+3]-self.network[j,0+3])**2 + (self.network[i,1+3]-self.network[j,1+3])**2 + (self.network[i,2+3]-self.network[j,2+3])**2
		b_uu = (self.network[i,u+3]-self.network[j,u+3]) * (self.network[i,u+3]-self.network[j,u+3])
		b_vv = (self.network[i,v+3]-self.network[j,v+3]) * (self.network[i,v+3]-self.network[j,v+3])
		k_uu = 4 * (self.epsilon**2) * self.beta * np.power((1 + (self.epsilon**2)*metric), self.beta - 1)
		k_vv = 4 * (self.epsilon**4) * self.beta * (self.beta - 1) * (b_uu + b_vv) * np.power((1 + (self.epsilon**2)*metric), self.beta - 2)
		return -(k_uu + k_vv)
	
	def kernel_uv(self, i, Bx, By, Bz, u, v):
		metric = (self.network[i,0+3] - Bx)**2 + (self.network[i,1+3] - By)**2 + (self.network[i,2+3] - Bz)**2
		if (u==3 and v==4):
			del_coeff = (self.network[i,u] - Bx) * (self.network[i,v] - By)
		elif (u==3 and v==5):
			del_coeff = (self.network[i,u] - Bx) * (self.network[i,v] - Bz)
		elif (u==4 and v==5):
			del_coeff = (self.network[i,u] - By) * (self.network[i,v] - Bz)
		return 4 * (self.epsilon**4) * self.beta * (self.beta - 1) * del_coeff * np.power((1 + self.epsilon*metric), self.beta - 2)

	def kernel_uu(self, i, Bx, By, Bz, u, v):
		metric = (self.network[i,0+3] - Bx)**2 + (self.network[i,1+3] - By)**2 + (self.network[i,2+3] - Bz)**2
		if (u==3 and v==4):
			b_uu = (self.network[i,u] - Bx) * (self.network[i,u] - Bx)
			b_vv = (self.network[i,v] - By) * (self.network[i,v] - By)
		elif (u==3 and v==5):
			b_uu = (self.network[i,u] - Bx) * (self.network[i,u] - Bx)
			b_vv = (self.network[i,v] - Bz) * (self.network[i,v] - Bz)
		elif (u==4 and v==5):
			b_uu = (self.network[i,u] - By) * (self.network[i,u] - By)
			b_vv = (self.network[i,v] - Bz) * (self.network[i,v] - Bz)
		k_uu = 4 * (self.epsilon**2) * self.beta * np.power((1 + (self.epsilon**2)*metric), self.beta - 1)
		k_vv = 4 * (self.epsilon**4) * self.beta * (self.beta - 1) * (b_uu + b_vv) * np.power((1 + (self.epsilon**2)*metric), self.beta - 2)
		return -(k_uu + k_vv)
	
	def Mag_Vector(self, Bx, By, Bz):
		self.mv_11 = [self.kernel_uu(i, Bx=Bx, By=By, Bz=Bz, u=4, v=5) for i in range(self.N)]
		self.mv_12 = [self.kernel_uv(i, Bx=Bx, By=By, Bz=Bz, u=3, v=4) for i in range(self.N)]
		self.mv_13 = [self.kernel_uv(i, Bx=Bx, By=By, Bz=Bz, u=3, v=5) for i in range(self.N)]
		self.mv_21 = [self.kernel_uv(i, Bx=Bx, By=By, Bz=Bz, u=3, v=4) for i in range(self.N)]
		self.mv_22 = [self.kernel_uu(i, Bx=Bx, By=By, Bz=Bz, u=3, v=5) for i in range(self.N)]
		self.mv_23 = [self.kernel_uv(i, Bx=Bx, By=By, Bz=Bz, u=4, v=5) for i in range(self.N)]
		self.mv_31 = [self.kernel_uv(i, Bx=Bx, By=By, Bz=Bz, u=3, v=5) for i in range(self.N)]
		self.mv_32 = [self.kernel_uv(i, Bx=Bx, By=By, Bz=Bz, u=4, v=5) for i in range(self.N)]
		self.mv_33 = [self.kernel_uu(i, Bx=Bx, By=By, Bz=Bz, u=3, v=4) for i in range(self.N)]
		self.mv_1 = np.column_stack((self.mv_11, self.mv_12, self.mv_13))
		self.mv_2 = np.column_stack((self.mv_21, self.mv_22, self.mv_23))
		self.mv_3 = np.column_stack((self.mv_31, self.mv_32, self.mv_33))
		return np.vstack((self.mv_1, self.mv_2, self.mv_3))

	def A_11(self):
		self.a_11 = [self.basis_uu(i,j, u=1,v=2) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_11).reshape(self.N, self.N)

	def A_12(self):
		self.a_12 = [self.basis_uv(i,j, u=0,v=1) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_12).reshape(self.N, self.N)

	def A_13(self):
		self.a_13 = [self.basis_uv(i,j, u=0,v=2) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_13).reshape(self.N, self.N)

	def A_21(self):
		self.a_21 = [self.basis_uv(i,j, u=1,v=0) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_21).reshape(self.N, self.N)

	def A_22(self):
		self.a_22 = [self.basis_uu(i,j, u=0,v=2) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_22).reshape(self.N, self.N)

	def A_23(self):
		self.a_23 = [self.basis_uv(i,j, u=1,v=2) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_23).reshape(self.N, self.N)

	def A_31(self):
		self.a_31 = [self.basis_uv(i,j, u=2,v=0) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_31).reshape(self.N, self.N)

	def A_32(self):
		self.a_32 = [self.basis_uv(i,j, u=2,v=1) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_32).reshape(self.N, self.N)

	def A_33(self):
		self.a_33 = [self.basis_uu(i,j, u=0,v=1) for i in range(self.N) for j in range(self.N)]
		return np.array(self.a_33).reshape(self.N, self.N)

	def A(self):
		self.A_1 = np.vstack((self.A_11(), self.A_12(), self.A_13()))
		self.A_2 = np.vstack((self.A_21(), self.A_22(), self.A_23()))
		self.A_3 = np.vstack((self.A_31(), self.A_32(), self.A_33()))
		return np.column_stack((self.A_1, self.A_2, self.A_3))
	
	def W(self, Bx, By, Bz):
		lon = self.network[:,0].reshape(self.N, 1)
		lat = self.network[:,1].reshape(self.N, 1)
		alt = self.network[:,2].reshape(self.N, 1)
		GPS = np.vstack((lon,lat,alt))
		return np.dot(np.linalg.inv(self.A()), GPS)

#===========================================================================================================================================================
# Kriging Algorithm (Gaussian regression process) Geostatistical Estimator
class OKI(object):

	def __init__(self, MPS_DATA_FILE='/home/joe/Desktop/Project_MPS37/IGRF_Data/fl_data.txt', EPSILON=1e12, BETA=0.00000001):
		self.network = np.loadtxt(MPS_DATA_FILE)
		self.epsilon = EPSILON
		self.beta = BETA
		self.N = self.network.shape[0]

	def covariance(self, i, j):
		self.metric = (self.network[i,0+3]-self.network[j,0+3])**2 + (self.network[i,1+3]-self.network[j,1+3])**2 + (self.network[i,2+3]-self.network[j,2+3])**2
		return np.power((1 + self.epsilon*self.metric), self.beta)

	def covector(self, Bx, By, Bz, i):
		
		return np.power((self.network[i,0+3] - Bx)**2 + (self.network[i,0+4] - By)**2 + (self.network[i,0+5] - Bz)**2, self.beta)

	def CM_solve(self):
		CoMatrix = np.matrix([self.covariance(i,j) for i in range(self.N) for j in range(self.N)]).reshape(self.N,self.N)
		CM_side = np.ones((self.N, 1))
		CM_corner = np.zeros((1, 1))
		CM_base = np.ones((1, self.N))
		CM_body = np.vstack((CoMatrix, CM_base))
		CM_left = np.vstack((CM_side, CM_corner))
		COM = np.column_stack((CM_body, CM_left))
		return np.linalg.inv(COM)

	def interpolate(self, Bx, By, Bz, GPS, pred_variance=False):
		CoVector = np.matrix([self.covector(Bx,By,Bz,i) for i in range(self.N)]).T 
		COV = np.vstack((CoVector,1))
		weight = np.dot(self.CM_solve(), COV)
		weight_1 = np.delete(weight, (-1), axis=0)
		lagrange_mult = weight[-1]
		prediction = np.dot(weight_1.T, self.network[:,GPS])
		variance = np.dot(weight.T, COV)
		if (pred_variance==True):
			return float(prediction), float(variance)
		elif (pred_variance==False):
			return float(prediction)

	def geo_map(self, lon, lat):
		plt.hold(True)
		plt.plot(-82.40286, 27.50134, 'bs')
		plt.plot(lon, lat, 'rs')
		plt.show()

#============================================================================================================================================================
if __name__ == "__main__":

	oki = OKI()
	test_LON = oki.interpolate(Bx=Bx_0, By=By_0, Bz=Bz_0, GPS=0, pred_variance=False)
	test_LAT = oki.interpolate(Bx=Bx_0, By=By_0, Bz=Bz_0, GPS=1, pred_variance=False)
	test_ALT = oki.interpolate(Bx=Bx_0, By=By_0, Bz=Bz_0, GPS=2, pred_variance=False)	
	oki.geo_map(test_LON, test_LAT)
	print '\nGPS Estimation:'
	print 'LAT = ', test_LAT
	print 'LON = ', test_LON
	print 'ALT = ', test_ALT
	print '\nPass: project_mps (v4.2.1)'

#============================================================================================================================================================	













