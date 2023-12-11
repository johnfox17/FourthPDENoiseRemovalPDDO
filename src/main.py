
import cv2
import numpy as np
import fourthOrderPDDODiscretization as PDDO4
import paperDiscretization
import createMesh

def main():
    lena = cv2.imread('../data/lena.png')
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    noisyLena = cv2.imread('../data/noisyLena.png')
    noisyLena = cv2.cvtColor(noisyLena, cv2.COLOR_BGR2GRAY)

    
    #mesh = createMesh.createMesh()
    #mesh.solve()

    paperMethod = paperDiscretization.paperDiscretization(noisyLena)
    paperMethod.solve()

    #PDDOMethod = PDDO4.fourthOrderPDDODiscretization()
    #PDDOMethod.solve()

    
    
    #np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\FourthPDENoiseRemovalPDDO\\data\\PDDOKernelMesh4.csv', PDDOMethod4.PDDOKernelMesh, delimiter=",")
    #cv2.imshow('Lean vs Noisy Lena',np.concatenate((lena, noisyLena), axis=1))

    #cv2.imshow('PDDO Edge Detection', filterIM1.filteredImage)
    

    #cv2.imshow('Noisy Lena',noisyLena)
    #cv2.waitKey(0)
    #print(np.shape(noisyLena))






    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
