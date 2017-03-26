#Jonathan Curtis
#This is a program for simulating the 1-D Ising model
#PHYS603 Assignment 8 

import numpy as np 

class Lattice:
	"""This class is for a 1-D lattice of Ising spins"""
	def __init__(self,Size,J,H):
		"""Initializes a lattice of size 'size' with all spins up and constants J,H"""
		self.size = Size
		self.jConst = J		#The lattice spin-spin coupling
		self.hConst = H		#The lattice magnetization constant

		#We represent the spins as 
		#down = -1
		#up = +1
		#We initialize a new configuation in the 'all-up' configuration
		self.spins = np.full(shape=self.size,fill_value=1.0)
		
		#We now create variables to store the lattice magnetization, pairing, and energy
		#These are defined by 
		#magPerSite = 1/L sum_j S[j]
		#pairPerSite = 1/L sum_j S[j]S[j+1]
		#energyPerSite = h*magPerSite + J*pairPerSite
		#These are all initialized to their value when all the spins are down.
		#They are updated every metropolis step or when randomized
		#When all spins are down the magPerSite is 1.0
		#When all spins are down the pairPerSite is +1.0
		self.magPerSite = 1.0
		self.pairPerSite = 1.0
		self.energyPerSite = -self.hConst - self.jConst

	def __repr__(self):
		return self.spins.__repr__()
	
	def calcEnergy(self):
		"""Computes the lattice-site-average magnetization, pairing, and energy of the lattice"""
		#Magnetization = 1/N Sum_j S[j]
		self.magPerSite = np.sum(self.spins)/np.float(self.size)
	
		#Pairing = 1/N Sum_j S[j] S[j-1]
		self.pairPerSite = np.sum(self.spins[:]*self.spins[:-1])/np.float(self.size)

		#Energy = -h*mag - J*pairing
		self.energyPerSite = -self.hConst*self.magPerSite - self.jConst*self.pairPerSite

	def randomize(self):
		"""Randomizes the spins in the lattice with a uniform distribution"""
		self.spins = np.random.randint(np.array([-1.0,1.0]),size=self.size)
		self.calcEnergy()

	def metropolisStep(self):
		"""This method implements the metropolis algorithm for flipping a spin"""
		"""First it randomly selects a site on the lattice"""
		"""Then it flips the spin with a probability related to the difference in energies"""
		
		#First we pick a random site
		#Then we retrieve its value and the value of its neighbors
		#They are put into an array as [site-1, site, site+1]
		#Note the mod % for the implementation of periodic boundary conditions
		site = np.random.randint(0,self.size)
		neighbors = np.array([self.spins[site-1],self.spins[site],self.spins[(site+1)%(self.size)]])	
		
		#Now we calculate the change in energy we would get by flipping this spin 
		#We also compute the change in magnetization and pairing 
		pairChange = -2.0*neighbors[1]*(neighbors[0]+neighbors[2])
		magChange = -2.0*neighbors[1]
		energyChange = -self.hConst*magChange - self.jConst*pairChange	

		#Now we accept the spin flip with the probability of min(1,e^{-Delta E})
		#To do this, we draw a number r randomly from Uniform(0,1)
		#We accept the flip if r < exp(-energyChange)
		#That way, if the energy is lowered energyChange<0 and this will always be accepted since exp(-energyChange) >=1
		#If the energy is raised, it will be accepted with probability 0<exp(-energyChange)<1
		#If we accept the change, we will also update the magnetization and energy of the lattice by adding the changes 
		#It also returns whether the flip was accepted or rejected 
		testnum = np.random.ranf()

		if testnum < np.exp(-energyChange):
			self.spins[site] *= -1.0	#This flips the spin value 
			
			#Now we update the internal variables
			self.energyPerSite += energyChange/float(self.size)
			self.pairPerSite += pairChange/float(self.size)
			self.magPerSite += magChange/float(self.size)
		
			return 1
		else:
			return 0

def main():
	print("No Test Code")	

if __name__ == "__main__":
	main()

