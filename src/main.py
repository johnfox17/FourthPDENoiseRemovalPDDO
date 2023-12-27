
import cv2
import numpy as np
import fourthOrderPDDODiscretization as PDDO4
import paperDiscretization
import PDDODenoising
import calcMetrics

def main():
    lena = cv2.imread('../data/lena.png')
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    noisyLena = cv2.imread('../data/noisyLena.png')
    noisyLena = cv2.cvtColor(noisyLena, cv2.COLOR_BGR2GRAY)

    

    #np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\lenaBefore.csv', noisyLena, delimiter=",")

    paperMethod = paperDiscretization.paperDiscretization(noisyLena)
    paperMethod.solve()
    #np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\denoisedDespeckledImages_Paper.csv', paperMethod.denoisedDespeckledImages, delimiter=",")
    np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/denoisedLena_Paper.csv', paperMethod.denoisedDespeckledImages, delimiter=",")

    PDDOMethod = PDDODenoising.PDDODenoising(noisyLena)
    PDDOMethod.solve()
    #np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\denoisedDespeckledImages_PDDO.csv', PDDOMethod.denoisedDespeckledImages, delimiter=",")
    np.savetxt('/home/doctajfox/Documents/Thesis/FourthPDENoiseRemovalPDDO/data/denoisedLena_PDDO.csv', PDDOMethod.noisyLena, delimiter=",") 
    
    #np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\PDDOKernelMesh4.csv', PDDOMethod4.PDDOKernelMesh, delimiter=",")
    metrics = calcMetrics.calcMetrics(lena, paperMethod.denoisedDespeckledImages, PDDOMethod.noisyLena)
    metrics.solve()
    print(metrics.paperPSNR)
    print(metrics.pddoPSNR)
    #cv2.imshow('Lean vs Noisy Lena',np.concatenate((lena, noisyLena), axis=1))

    #cv2.imshow('PDDO Edge Detection', filterIM1.filteredImage)
    

    #cv2.imshow('Noisy Lena',noisyLena)
    #cv2.waitKey(0)
    #print(np.shape(noisyLena))






    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
