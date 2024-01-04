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
        self.laplacianOfIntensity = signal.convolve2d(self.noisyLena, self.PDDOLaplacianKernel, boundary='symm', mode='same')*self.dx*self.dy

    def calcCoefficients(self):
        self.coefficients = np.zeros((self.numRows, self.numColumns))
        k = 0.05
        for iRow in range(self.numRows):
            for iColumn in range(self.numColumns):
                self.coefficients[iRow, iColumn] = 1/(1+ np.abs(self.laplacianOfIntensity[iRow,iColumn]/k)**2)
    
    def calcLaplacianOfCoefficients(self):
        self.laplacianOfCoefficients = signal.convolve2d(self.coefficients, self.PDDOLaplacianKernel, boundary='symm', mode='same')*self.dx*self.dy

    def calcLaplacianOfLaplacianOfIntensity(self):
        self.laplacianOfLaplacianOfIntensity = signal.convolve2d(self.laplacianOfIntensity, self.PDDOLaplacianKernel, boundary='symm', mode='same')*self.dx*self.dy

    def padImage(self):
        self.noisyLena = np.pad(self.noisyLena, ((1, 1), (1, 1)), 'symmetric')
    
    def despeckleImage(self):
        k = 3.0
        self.despeckledImage = np.zeros((self.numRows+2, self.numColumns+2))
        for iRow in range(1,self.numRows+1,1):
            for iColumn in range(1,self.numColumns+1,1):
                currentMean = (self.noisyLena[iRow+1,iColumn] + self.noisyLena[iRow-1,iColumn] + self.noisyLena[iRow,iColumn+1] + self.noisyLena[iRow,iColumn-1])/4
                currentSigSqr = (self.noisyLena[iRow+1,iColumn]**2 + self.noisyLena[iRow-1,iColumn]**2 + self.noisyLena[iRow,iColumn+1]**2 + self.noisyLena[iRow,iColumn-1]**2)/4 - currentMean**2
                if ((np.abs(self.noisyLena[iRow,iColumn] - currentMean))**2 > k*currentSigSqr):
                    self.despeckledImage[iRow,iColumn] = currentMean
                else:
                    self.despeckledImage[iRow,iColumn]  = self.noisyLena[iRow,iColumn]
        self.noisyLena = self.despeckledImage[1:self.numRows+1,1:self.numColumns+1]


    def solve(self):
        self.createPDDOKernelMesh()
        PDDOLaplacian = calcPDDOLaplacian.calcPDDOLaplacian(self.PDDOKernelMesh)
        PDDOLaplacian.solve()
        self.PDDOLaplacianKernel = PDDOLaplacian.kernel
        for iTime in range(1000):
            self.calcLaplacianOfIntensity()
            self.calcCoefficients()
            self.calcLaplacianOfCoefficients()
            self.calcLaplacianOfLaplacianOfIntensity()
            self.noisyLena = self.noisyLena - 0.25*(np.multiply(self.coefficients, self.laplacianOfLaplacianOfIntensity) + np.multiply(self.laplacianOfCoefficients, self.laplacianOfIntensity))
        
        self.padImage()
        self.despeckleImage()



