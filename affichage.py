import cv2

#Affichage des nombres sur l'image finale
def affichagenombres(image, grille):
    #the step is a will help to find a location
    step_x = image.shape[1] // 9
    step_y = image.shape[0] // 9

    font = cv2.FONT_HERSHEY_SIMPLEX

    for x in range(0, 9):
        for y in range(0, 9):
            cv2.putText(image, str(grille[y][x]), (x*step_x+18, y*step_y+53), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    return image