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
    img = cv2.imread('sudokuboard1.png')

    #will create a top down view of the grid
    transformed = findsudokugrid.getGridContour(img)


    #create the CNN network if it doesn't exists
    if(os.path.exists("model_train.h5")):
        model = tf.keras.models.load_model("model_train.h5")
    else:
        model = digit_recognizer.build()

    
    grille = solver.createList(transformed, model)


    if (Suduko(grille, 0, 0)):
        resultat = puzzle(grille)
    else:
        print("Cette grille n'a pas de solution")
    
    print(resultat)
    
    #affichge sur une image
    imageFinale = affichage.affichagenombres(transformed, resultat)

    cv2.imshow("Resultat : ", imageFinale)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()