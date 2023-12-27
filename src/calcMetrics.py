import numpy as np

class calcMetrics:
    def __init__(self, originalLena, paperMethod, PDDOMethod):
        self.originalLena = originalLena
        self.paperMethod = paperMethod
        self.PDDOMethod = PDDOMethod
        self.numRows, self.numColumns = np.shape(self.originalLena)

    def PSNR(self):
        self.paperPSNR = 10*np.log10(255**2/(np.linalg.norm(self.originalLena.flatten()-self.paperMethod.flatten()))**2)
        self.pddoPSNR = 10*np.log10(255**2/(np.linalg.norm(self.originalLena.flatten()-self.PDDOMethod.flatten()))**2)


    def solve(self):
        self.PSNR()

