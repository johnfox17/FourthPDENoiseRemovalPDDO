import PDDOConstants
import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve

class calcPDDOLaplacian:
    def __init__(self, PDDOKernelMesh):
        self.dx = 1/PDDOConstants.NX
        self.dy = 1/PDDOConstants.NY
        self.deltaX = PDDOConstants.HORIZON2*self.dx
        self.deltaY = PDDOConstants.HORIZON2*self.dy
        self.bVec20 = PDDOConstants.BVEC20 
        self.bVec02 = PDDOConstants.BVEC02
        self.horizon = PDDOConstants.HORIZON2
        self.kernelDim = PDDOConstants.KERNELDIM2
        self.PDDOKernelMesh = PDDOKernelMesh


    def calculateXis(self):
        midPDDONodeCoords = self.PDDOKernelMesh[int((len(self.PDDOKernelMesh)-1)/2),:]
        self.xXis = midPDDONodeCoords[0]-self.PDDOKernelMesh[:,0]
        self.yXis = midPDDONodeCoords[1]-self.PDDOKernelMesh[:,1]
         
    def calculateGPolynomials(self):
        deltaMag = np.sqrt(self.deltaX**2 + self.deltaY**2)
        diffMat = np.zeros([6,6])
        g20 = []
        g02 = []

        for iNode in range(len(self.PDDOKernelMesh)):
            currentXXi = self.xXis[iNode]
            currentYXi = self.yXis[iNode]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2)
            pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, (currentXXi/deltaMag)**2, \
                    (currentYXi/deltaMag)**2, (currentXXi/deltaMag)*(currentYXi/deltaMag)])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            diffMat += weight*np.outer(pList,pList)*self.dx*self.dy
        for iNode in range(len(self.PDDOKernelMesh)):
            currentXXi = self.xXis[iNode]
            currentYXi = self.yXis[iNode]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2)
            pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, (currentXXi/deltaMag)**2, \
                    (currentYXi/deltaMag)**2, (currentXXi/deltaMag)*(currentYXi/deltaMag)])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            g20.append(weight*(np.inner(solve(diffMat,self.bVec20), pList)))
            g02.append(weight*(np.inner(solve(diffMat,self.bVec02), pList)))
        self.g20 = np.array(g20).reshape((self.kernelDim,self.kernelDim))
        self.g02 = np.array(g02).reshape((self.kernelDim,self.kernelDim))
    
    def combineGPolynomials(self):
        self.kernel = self.g20 + self.g02

    def solve(self):
        self.calculateXis()
        self.calculateGPolynomials()
        self.combineGPolynomials()
