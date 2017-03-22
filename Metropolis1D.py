#Jonathan Curtis
#This is a program for simulating the 1-D Ising model
#PHYS603 Assignment 8 

import numpy as np 
import random as rnd

def scaleToSymmInt(x):
	"""Rescales x in [0,1] to x in [-1,1] symmetrically"""
	return 2.0*(x-.5)

class Lattice:
	"""This class is for a 1-D lattice of Ising spins"""
	def __init__(self, Size,J,H):
		"""Initializes a lattice of size 'size' with all spins down"""
		self.size = Size
		self.jConst = J		#The lattice spin-spin coupling
		self.hConst = H		#The lattice magnetization constant

		#We represent the spins as 
		#down = 0
		#up = 1
		self.spins = np.zeros(self.size)

		#Computes the lattice-site average magnetization of the configuation
		#Initialized to zero, updated when appropriate methods are called 
		self.magPerSite = 0.0 
		
		#Computes the lattice-site average pairing interaction of the configuation
		#Initialized to zero, updated when appropriate methods are called 
		self.pairPerSite = 0.0

		#Compute the lattice-site average of the total energy of the configuration
		#Initialized to zero, updated when appropriate methods are called 
		self.energyPerSite = 0.0

	def __repr__(self):
		return self.spins.__repr__()

	def randomize(self):
		"""Randomizes the spins in the lattice with a uniform distribution"""
		self.spins = np.random.randint(low=0,high=2,size=self.size)

	def calcEnergy(self):
		"""Computes the lattice-site-average magnetization, pairing, and energy of the lattice"""
		#Magnetization = 1/N Sum_j S[j]
		self.magPerSite = scaleToSymmInt( np.float(np.sum(self.spins))/np.float(self.size))
	
		#Pairing = 1/N Sum_j S[j] S[j-1]
		pairInt = 0.0
		for i in np.arange(0,self.size):
			pairInt += scaleToSymmInt(self.spins[i])*scaleToSymmInt(self.spins[i-1])

		pairInt /= np.float(self.size)

		self.pairPerSite = pairInt 

		#Energy = h*mag + J*pairing
		self.energyPerSite = self.hConst*self.magPerSite + self.jConst*self.pairPerSite


	def metropolis(self):
		"""This method implements the metropolis algorithm for flipping a spin"""
		"""First it randomly selects a site on the lattice"""
		"""Then it flips the spin with a probability related to the difference in energies"""
		
		#First we pick a random site
		#Then we retrieve its value and the value of its neighbors
		site = np.random.randint(0,self.size)
		spin = self.spins[site]
		
		#They are put into an array as [site-1, site, site+1]
		neighbors = np.array([self.spins[site-1],spin,self.spins[(site+1)%(self.size)]])	
		#Note the mod % for the implementation of periodic boundary conditions
		
		#We rescale all spins to the appropriate range
		neighbors[:]= scaleToSymmInt(neighbors[:])

		#Now we calculate the change in energy we would get by flipping this spin 
		#We also compute the change in magnetization and pairing 
		#For a single spin-1/2 Ising spin being flipped the change in energy for flipping S[j] can be computed as 
		energyChange = -2.0*neighbors[1]*(self.hConst + self.jConst*(neighbors[0]+neighbors[2]))	
		pairChange = -2.0*neighbors[1]*(neighbors[0]+neighbors[2])
		magChange = -2.0*neighbors[1]

		#Now we accept the spin flip with the probability of min(1,e^{-Delta E})
		#To do this, we draw a number r randomly from Uniform(0,1)
		#We accept the flip if r < e^{-DeltaE}
		#That way, if the energy is lowered $DeltaE<0$ and this will always be accepted 
		#If the energy is raised, it will be accepted with probability $e^{-Delta E}<1$.
		testnum = np.random.ranf()

		#If we accept the change, we will also update the magnetization and energy of the lattice by adding the changes 
		if testnum < np.exp(-energyChange):
			self.spins[site] = 1 - spin	#This flips the spin value 
			
			#Now we update the internal variables
			self.energyPerSite += energyChange/self.size
			self.pairPerSite += pairChange/self.size
			self.magPerSite += magChange/self.size

def main():
	lattice = Lattice(10,1.0,1.0)
	lattice.randomize()
	print(lattice)

	lattice.calcEnergy()
	print(lattice.magPerSite)
	print(lattice.pairPerSite)
	print(lattice.energyPerSite)

	lattice.metropolis()
	print(lattice.magPerSite)
	print(lattice.pairPerSite)
	print(lattice.energyPerSite)
	

if __name__ == "__main__":
	main()

