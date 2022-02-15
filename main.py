import findsudokugrid
import solver
import digit_recognizer
import tensorflow as tf
import cv2
from solver import puzzle, Suduko
import affichage
import os

def main():
    #load the image
    img = cv2.imread('sudokuboard2.png')

    #will create a top down view of the grid
    transformed = findsudokugrid.getGridContour(img)

    #transformed = digit_recognizer.preProccess(transformed)
    grayscaled = cv2.cvtColor(transformed,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)


    # cv2.imshow("image", thresh)
    # cv2.waitKey(0)


    #create the CNN network if it doesn't exists
    if(os.path.exists("model_train.h5")):
        model = tf.keras.models.load_model("model_train.h5")
    else:
        model = digit_recognizer.build()

    
    grille = solver.createList(thresh, model)






    if (Suduko(grille, 0, 0)):
        resultat = puzzle(grille)
        print(resultat)
        #affichge sur une image
        imageFinale = affichage.affichagenombres(transformed, resultat)

        cv2.imshow("Resultat : ", imageFinale)
        cv2.waitKey(0)

    else:
        print("Cette grille n'a pas de solution")
        print(grille)
    
    


if __name__ == "__main__":
    main()