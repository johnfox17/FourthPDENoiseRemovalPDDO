import PDDOConstants
import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve
import calcPDDOLaplacian

class PDDODenoising:
    def __init__(self, noisyLena):
        self.noisyLena = noisyLena
        self.numRows, self.numColumns = np.shape(self.noisyLena)
        self.dx = 1/PDDOConstants.NX
        self.dy = 1/PDDOConstants.NY
        self.horizon = PDDOConstants.HORIZON2

    def createPDDOKernelMesh(self):
        indexing = 'xy'
        xCoords = np.arange(self.dx/2, (self.horizon*2 + 1)*self.dx, self.dx)
        yCoords = np.arange(self.dy/2, (self.horizon*2 + 1)*self.dy, self.dy)
        xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
        xCoords = xCoords.reshape(-1, 1)
        yCoords = yCoords.reshape(-1, 1)
        self.PDDOKernelMesh = np.array([xCoords[:,0], yCoords[:,0]]).T

    def applyBoundaryConditions(self):
        self.denoisedLena = np.pad(self.noisyLena, 1, mode='constant')
        # Left BC
        self.denoisedLena[1:self.numRows+1,0] = self.noisyLena[:,0]
        #Right BC
        self.denoisedLena[1:self.numRows+1,self.numColumns+1] = self.noisyLena[:,self.numColumns-1]
        #Top BC
        self.denoisedLena[self.numRows+1,1:self.numColumns+1] = self.noisyLena[self.numRows-1,:]
        #Bottom BC
        self.denoisedLena[0,1:self.numColumns+1] = self.noisyLena[0,:]
        #Corners
        self.denoisedLena[-1,0] = self.noisyLena[-1,0]
        self.denoisedLena[0,0] = self.noisyLena[0,0]
        self.denoisedLena[0,-1] = self.noisyLena[0,-1]
        self.denoisedLena[-1,-1] = self.noisyLena[-1,-1]
        self.denoisedLena = self.denoisedLena.astype(float)

    #def calcLaplacianOfIntensity(self):

    def solve(self):
        self.createPDDOKernelMesh()
        PDDOLaplacian = calcPDDOLaplacian.calcPDDOLaplacian(self.PDDOKernelMesh)
        PDDOLaplacian.solve()
        self.PDDOLaplacianKernel = PDDOLaplacian.kernel
        self.applyBoundaryConditions()
        #self.calcLaplacianOfIntensity()

        np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/lenaBC.csv', self.denoisedLena, delimiter=",",fmt='%3.3f')


