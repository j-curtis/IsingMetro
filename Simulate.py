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
		
class Ensemble:
	"""This is a class which generates an ensemble of simulations all starting from the same ordered state"""
	def __init__(self,Dimension,Size,J,H,NSteps,NTrials):
		"""Runs NTrials simulations of the other parameters"""
		self.dim = Dimension
		self.nTrials = NTrials 
		self.nSteps = NSteps 
		self.size = Size 
		self.jConst = J 
		self.hConst = H 

		#ensemble of energy, magnetization, and pair correlation vs. time
		self.energy = np.full(shape = (self.nTrials,self.nSteps), fill_value = 0.0)
		self.mag = np.full(shape = (self.nTrials,self.nSteps), fill_value = 0.0)
		self.pair = np.full(shape = (self.nTrials,self.nSteps), fill_value = 0.0)

		#ensemble averages of single-time observables vs. time
		#< E(t) >
		#And similarly for the other observables 
		self.energyMean = np.full(shape = self.nSteps, fill_value = 0.0)
		self.magMean = np.full(shape = self.nSteps, fill_value = 0.0)
		self.pairMean = np.full(shape = self.nSteps, fill_value = 0.0)

		#ensemble averages of two-time observables vs. time 
		# Sum_t( E(t) E(t+T) ) 
		self.energyCorr = np.full(shape = (self.nTrials,self.nSteps), fill_value = 0.0)
	
	def generate(self):
		"""Generates the data via simulations"""
		for i in np.arange(self.nTrials):
			sim = Simulation(self.dim,self.size,self.jConst,self.hConst,self.nSteps)
			sim.simulate()
			
			self.energy[i,:] += sim.energy[:] 
			self.mag[i,:] += sim.mag[:]
			self.pair[i,:] += sim.pair[:]

	def avg(self):
		"""Computes ensemble averages"""
		 
		self.energyMean = np.average(self.energy,axis=0)
		self.magMean = np.average(self.mag,axis=0)
		self.pairMean = np.average(self.pair,axis=0)

		#Correlation of energy 
		#Given by < Sum_t E(t) E(t+T)> = C(T)
		self.energyCorr = np.corrcoef(self.energy,rowvar=0)



