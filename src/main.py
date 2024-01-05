
import cv2
import numpy as np
import fourthOrderPDDODiscretization as PDDO4
import paperDiscretization
import PDDODenoising
import calcMetrics
import time

def main():
    lena = cv2.imread('../data/lena.png')
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    noisyLena = cv2.imread('../data/noisyLena.png')
    noisyLena = cv2.cvtColor(noisyLena, cv2.COLOR_BGR2GRAY)

    

    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\lenaBefore.csv', noisyLena, delimiter=",")

    t = time.time()
    paperMethod = paperDiscretization.paperDiscretization(noisyLena)
    paperMethod.solve()
    print('Time Elapsed Paper Method = ', time.time() - t)
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\denoisedLena_Paper.csv', paperMethod.noisyLena, delimiter=",")
    #np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/denoisedLena_Paper.csv', paperMethod.denoisedDespeckledImages, delimiter=",")'''
    t = time.time()
    PDDOMethod = PDDODenoising.PDDODenoising(noisyLena)
    PDDOMethod.solve()
    print('Time Elapsed PDDO Method = ', time.time() - t)
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\denoisedLena_PDDO.csv', PDDOMethod.noisyLena, delimiter=",")
    #np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/denoisedLena_PDDO.csv', PDDOMethod.noisyLena, delimiter=",") 
    







    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
