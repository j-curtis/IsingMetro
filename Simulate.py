#Jonathan Curtis
#This is a program for analyzing simulations of the Ising model
#PHYS603 Assignment 8 

import numpy as np
import Metropolis1D as m1d
import Metropolis2D as m2d

class Simulation:
	"""This is a class which runs a specific instance of a simulation of the 1D or 2D Ising model"""
	def __init__(self, Dimension, Size, J, H, NSteps):
		"""Initializes and runs a simulation with given lattice size and parameters for given time"""
		self.dim = Dimension

		if self.dim == 2:
			self.latt = m2d.Lattice(Size,J,H)
		else:
			self.latt = m1d.Lattice(Size,J,H)

		self.nSteps = NSteps
		
		#We create variables for observables vs. time
		#We store the energy, magnetization, and nearest-neighbor correlations (pair) vs. time 
		#We initialize the arrays to be full of the initial values of these observables
		self.energy = np.full(shape=self.nSteps,fill_value=self.latt.energyPerSite)	
		self.mag = np.full(shape=self.nSteps,fill_value=self.latt.magPerSite)		
		self.pair = np.full(self.nSteps,fill_value=self.latt.pairPerSite)
	
	def __repr__(self):
		return self.pair.__repr__() 

	def simulate(self):
		"""Runs the simulation and fills the data structures"""
		for i in np.arange(0,self.nSteps):
			#Fill observables
			self.energy[i] = self.latt.energyPerSite
			self.mag[i] = self.latt.magPerSite
			self.pair[i] = self.latt.pairPerSite

			#Run one time step
			self.latt.metropolisStep()
		
			
