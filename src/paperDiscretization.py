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

    def despeckleImage(self):
        k = 3.0
        self.despeckledImage = np.zeros((self.numRows+2, self.numColumns+2))
        for iRow in range(1,self.numRows+1,1):
            for iColumn in range(1,self.numColumns+1,1):
                currentMean = (self.denoisedLena[iRow+1,iColumn] + self.denoisedLena[iRow-1,iColumn] + self.denoisedLena[iRow,iColumn+1] + self.denoisedLena[iRow,iColumn-1])/4
                currentSigSqr = (self.denoisedLena[iRow+1,iColumn]**2 + self.denoisedLena[iRow-1,iColumn]**2 + self.denoisedLena[iRow,iColumn+1]**2 + self.denoisedLena[iRow,iColumn-1]**2)/4 - currentMean**2
                if ((np.abs(self.denoisedLena[iRow,iColumn] - currentMean))**2 > k*currentSigSqr):
                    self.despeckledImage[iRow,iColumn] = currentMean
                else:
                    self.despeckledImage[iRow,iColumn]  = self.denoisedLena[iRow,iColumn]

    def timeIntegrate(self):
        numTimeSteps = 1000
        denoisedImages = []
        denoisedDespeckledImages = []
        for i in range(numTimeSteps):
            self.applyBoundaryConditions()
            self.calculateLaplacianOfIntensity()
            self.denoisedLena =  np.array(self.laplacianOfIntensity[1:self.numRows+1,1:self.numColumns+1])
            self.applyBoundaryConditions()    
            self.laplacianOfGFunction = np.array(self.denoisedLena)
            self.calcCoefficients()
            self.calculateLaplacianOfGFunction()
            self.denoisedLena = self.denoisedLena - 0.25*self.laplacianOfGFunction
            self.noisyLena = np.array(self.denoisedLena[1:self.numRows+1,1:self.numColumns+1])
            self.despeckleImage()
        self.denoisedDespeckledImages = self.despeckledImage
             
    def solve(self):
        self.timeIntegrate()
