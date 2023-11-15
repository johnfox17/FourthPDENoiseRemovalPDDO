import numpy as np
#Defining constants for PDDO algorithm
L1 = 1
L2 = 1
NX = 512
NY = 512
HORIZON = 5.015 #Because the PDE is 4th order
BVEC40 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0])
BVEC04 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0])
