
import cv2
import numpy as np
import fourthOrderPDDODiscretization as PDDO



def main():
    lena = cv2.imread('../data/lena.png')
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    noisyLena = cv2.imread('../data/noisyLena.png')
    noisyLena = cv2.cvtColor(noisyLena, cv2.COLOR_BGR2GRAY)
    
    PDDOMethod = PDDO.fourthOrderPDDODiscretization()
    PDDOMethod.solve()

    cv2.imshow('Lean',lena) 
    cv2.imshow('Noisy Lena',noisyLena)
    cv2.waitKey(0)
    print(np.shape(noisyLena))






    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
