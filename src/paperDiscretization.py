import numpy as np


class paperDiscretization:
    def __init__(self, noisyLena):
        self.noisyLena = noisyLena
        self.numRows, self.numColumns = np.shape(self.noisyLena) 
        self.dx = 1
        self.dy = 1
        self.hsquared = self.dx*self.dy

    def applyBoundaryConditions(self):
        self.denoisedLena = np.pad(self.noisyLena, ((1, 1), (1, 1)), 'symmetric')
        self.denoisedLena = self.denoisedLena.astype(float) 
        
    def calculateLaplacianOfIntensity(self):
        self.laplacianOfIntensity = np.zeros((self.numRows+2, self.numColumns+2))
        for iRow in range(1,self.numRows+1,1):
            for iColumn in range(1,self.numColumns+1,1):
                self.laplacianOfIntensity[iRow,iColumn] = (self.denoisedLena[iRow+1,iColumn] + self.denoisedLena[iRow-1,iColumn] + self.denoisedLena[iRow,iColumn+1] + self.denoisedLena[iRow,iColumn-1] - 4 * self.denoisedLena[iRow,iColumn])/(self.hsquared)
        self.laplacianOfIntensity = self.laplacianOfIntensity[1:self.numRows+1,1:self.numColumns+1]
        
    def applyBoundaryConditionsToLaplacian(self):
        self.laplacianOfIntensity = np.pad(self.laplacianOfIntensity, ((1, 1), (1, 1)), 'symmetric')

    def calcCoefficients(self):
        self.coefficients = np.zeros((self.numRows+2, self.numColumns+2))
        k = 0.05
        for iRow in range(1,self.numRows+1,1):
            for iColumn in range(1,self.numColumns+1,1):
                self.coefficients[iRow, iColumn] = 1/(1+ (self.laplacianOfIntensity[iRow,iColumn]/k)**2)
        
    def calculateLaplacianOfGFunction(self):
        self.laplacianOfGFunction = np.zeros((self.numRows+2, self.numColumns+2))
        gFunction = np.multiply(self.coefficients, self.laplacianOfIntensity)
        for iRow in range(1,self.numRows+1,1):
            for iColumn in range(1,self.numColumns+1,1):
                self.laplacianOfGFunction[iRow,iColumn] = self.coefficients[iRow,iColumn]*((self.laplacianOfIntensity[iRow+1,iColumn] + self.laplacianOfIntensity[iRow-1,iColumn] + self.laplacianOfIntensity[iRow,iColumn+1] + self.laplacianOfIntensity[iRow,iColumn-1] - 4 * self.laplacianOfIntensity[iRow,iColumn])/self.hsquared) + self.laplacianOfIntensity[iRow,iColumn]*((self.coefficients[iRow+1,iColumn] + self.coefficients[iRow-1,iColumn] + self.coefficients[iRow,iColumn+1] + self.coefficients[iRow,iColumn-1] - 4 * self.coefficients[iRow,iColumn])/self.hsquared)

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
        numTimeSteps = 1000
        for i in range(numTimeSteps):
            self.applyBoundaryConditions()
            self.calculateLaplacianOfIntensity()
            self.applyBoundaryConditionsToLaplacian()    
            self.calcCoefficients()
            self.calculateLaplacianOfGFunction()
            self.denoisedLena = self.denoisedLena - 0.25*self.laplacianOfGFunction
            self.noisyLena = np.array(self.denoisedLena[1:self.numRows+1,1:self.numColumns+1])
        
        self.padImage()
        self.despeckleImage()
             
