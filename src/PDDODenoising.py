import PDDOConstants
import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve
import calcPDDOLaplacian
from scipy import signal

class PDDODenoising:
    def __init__(self, noisyLena):
        self.noisyLena = noisyLena
        self.numRows, self.numColumns = np.shape(self.noisyLena)
        self.dx = 1/PDDOConstants.NX
        self.dy = 1/PDDOConstants.NY
        self.horizon = PDDOConstants.HORIZON2
        self.deltaX = self.horizon*self.dx
        self.deltaY = self.horizon*self.dy
        self.deltaMag = np.sqrt(self.deltaX**2 + self.deltaY**2)

    def createPDDOKernelMesh(self):
        indexing = 'xy'
        xCoords = np.arange(self.dx/2, (self.horizon*2 + 1)*self.dx, self.dx)
        yCoords = np.arange(self.dy/2, (self.horizon*2 + 1)*self.dy, self.dy)
        xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
        xCoords = xCoords.reshape(-1, 1)
        yCoords = yCoords.reshape(-1, 1)
        self.PDDOKernelMesh = np.array([xCoords[:,0], yCoords[:,0]]).T

    def calcLaplacianOfIntensity(self):
        self.denoisedLena = signal.convolve2d(self.noisyLena, self.PDDOLaplacianKernel, boundary='symm', mode='same')*self.dx*self.dy

    def calcCoefficients(self):
        self.coefficients = np.zeros((self.numRows, self.numColumns))
        k = 0.1
        for iRow in range(self.numRows):
            for iColumn in range(self.numColumns):
                #print(self.denoisedLena[iRow,iColumn])
                self.coefficients[iRow, iColumn] = 1/(1+ np.abs(self.denoisedLena[iRow,iColumn]/k)**2)
    
    def solve(self):
        self.createPDDOKernelMesh()
        PDDOLaplacian = calcPDDOLaplacian.calcPDDOLaplacian(self.PDDOKernelMesh)
        PDDOLaplacian.solve()
        self.PDDOLaplacianKernel = PDDOLaplacian.kernel
        self.calcLaplacianOfIntensity()
        self.calcCoefficients()

        #np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/coefficients.csv', self.coefficients, delimiter=",",fmt='%3.3f')
        np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/laplacian.csv', self.denoisedLena, delimiter=",")


