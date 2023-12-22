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

    def createPDDOKernelMesh(self):
        indexing = 'xy'
        xCoords = np.arange(self.dx/2, (self.horizon*2 + 1)*self.dx, self.dx)
        yCoords = np.arange(self.dy/2, (self.horizon*2 + 1)*self.dy, self.dy)
        xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
        xCoords = xCoords.reshape(-1, 1)
        yCoords = yCoords.reshape(-1, 1)
        self.PDDOKernelMesh = np.array([xCoords[:,0], yCoords[:,0]]).T

    def calcLaplacianOfIntensity(self):
        self.denoisedLena = signal.convolve2d(self.noisyLena, self.PDDOLaplacianKernel, boundary='symm', mode='same')

    def solve(self):
        self.createPDDOKernelMesh()
        PDDOLaplacian = calcPDDOLaplacian.calcPDDOLaplacian(self.PDDOKernelMesh)
        PDDOLaplacian.solve()
        self.PDDOLaplacianKernel = PDDOLaplacian.kernel
        self.calcLaplacianOfIntensity()

        np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/laplacianImage.csv', self.denoisedLena, delimiter=",",fmt='%3.3f')


