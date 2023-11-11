import PDDOConstants
import numpy as np
from sklearn.neighbors import KDTree

class fourthOrderPDDODiscretization:
    def __init__(self):
        self.dx = 1/PDDOConstants.NX
        self.dy = 1/PDDOConstants.NY
        self.l1 = PDDOConstants.L1
        self.l2 = PDDOConstants.L2
        self.deltaX = PDDOConstants.HORIZON*self.dx
        self.deltaY = PDDOConstants.HORIZON*self.dy
    
    def createPDDOMesh(self):
        indexing = 'xy'
        xCoords = np.linspace(self.dx/2,self.l1+self.dx,PDDOConstants.NX)
        yCoords = np.linspace(self.dy/2,self.l2+self.dy,PDDOConstants.NY)
        xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
        xCoords = xCoords.reshape(-1, 1)
        yCoords = yCoords.reshape(-1, 1)
        self.coords = np.array([xCoords[:,0], yCoords[:,0]]).T

    def findFamilyMembers(self):
        tree = KDTree(self.coords, leaf_size=2)
        self.familyMembers = tree.query_radius(self.coords, r = self.deltaX)

    def solve(self):
        self.createPDDOMesh()
        self.findFamilyMembers()
        print(self.familyMembers) 
