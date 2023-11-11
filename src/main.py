
import cv2





def main():
    lena = cv2.imread('../data/lena.png')
    noisyLena = cv2.imread('../data/noisyLena.png')
    cv2.imshow('Lean',lena) 
    cv2.imshow('Noisy Lena',noisyLena)
    cv2.waitKey(0)






    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
