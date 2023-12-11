import numpy as np


class paperDiscretization:
    def __init__(self, noisyLena):
        self.noisyLena = noisyLena
        self.numRows, self.numColumns = np.shape(self.noisyLena) 

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
        
        np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\paddedImage.csv', self.denoisedLena, delimiter=",")
    
    '''def calculateLaplacianOfIntensity(self):
        for iRow in range(self.numRows):
            for iColumn in range(self.numColumns):
                if (iRow == 0):

                elif (iRow == self.numRows-1):

                elif (iColumn == 0):

                elif (iColumn == self.numColumns-1):
                
                else:
                print(self.noisyLena[iRow,iColumn])'''
    

    def timeIntegrate(self):
        self.applyBoundaryConditions()
        #self.calculateLaplacianOfIntensity()
    
    def solve(self):
        self.timeIntegrate()
