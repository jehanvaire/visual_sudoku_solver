import os
import findsudokugrid
import solver
import cv2
from solver import puzzle, Sudoku
import tensorflow as tf
import digit_recognizer


def main(filename, model):

    img = cv2.imread(filename)

    # will create a top down view of the grid
    transformed = findsudokugrid.getGridContour(img)

    # show image
    # cv2.imshow("image", transformed)


    gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # save the transformed image
    cv2.imwrite('transformed.jpg', thresh)

    # création de la grille de sudoku
    grille = solver.createList(thresh, model)


    if (Sudoku(grille, 0, 0)):
        resultat = puzzle(grille)
    else:
        resultat = grille
    
    print(resultat)

    return resultat

# if __name__ == '__main__':
#     # #create the CNN network if it doesn't exists
#     if(os.path.exists("model_train.h5")):
#         model = tf.keras.models.load_model("model_train.h5")
#     else:
#         model = digit_recognizer.build()

#     main("sudo.png", model)