import numpy as np
import findsudokugrid
import cv2
import digit_recognizer

#taille du sudoku
M=9

#will crop the image to 32*32, on its center
def cropImage(cell, taille):
    diff_hauteur = cell.shape[0] - taille
    diff_largeur = cell.shape[1] - taille
    crop = cell[int(diff_hauteur/2):taille+int(diff_hauteur/2), int(diff_largeur/2):taille+int(diff_largeur/2)]
    return crop

#will check if the image has a number 
def hasNumber(cell):

    # crop the image
    cell = cropImage(cell, cell.shape[0]-4)

    #will get the number of black pixels
    nb_black = 0
    for i in range(0, cell.shape[0]):
        for j in range(0, cell.shape[1]):
            if cell[i][j] == 0:
                nb_black += 1

    #if the number of black pixels is greater than 220
    if nb_black > 20:
        return True
    else:
        return False




def createList(transformed, model):
    grille = []

    #loop in all the cells in the grid
    for i in range(0, 9):
        #reset de tab
        tab = []
        for j in range(0, 9):
            #will get location of a specific cell
            cellLocation = findsudokugrid.findCellLocation(transformed, i, j)

            cell = findsudokugrid.getTopDownView(transformed, cellLocation)

            if(not hasNumber(cell)):
                nombre = 0
            else:
                cell = np.asarray(cell)
                cell = cv2.resize(cell, (32, 32))
                # cell = digit_recognizer.preProccess(cell)
                cell = cell.reshape(1, 32, 32, 1)

                
                prediction = model.predict(cell)[0]
                nombre = np.argmax(prediction)
                

            tab.append(nombre)
        grille.append(tab)
    return grille

# tests de création de grille
def createListDebug(transformed, model):
    cellLocation = findsudokugrid.findCellLocation(transformed, 0, 0)
    cell = findsudokugrid.getTopDownView(transformed, cellLocation)

    cv2.imshow("c", cell)
    cv2.waitKey(0)

    nombre = -1

    if(not hasNumber(cell)):
        nombre = 0
        print("No nombre")
    else:
        cell = np.asarray(cell)
        cell = cv2.resize(cell, (32, 32))
        # cell = digit_recognizer.preProccess(cell)
        cell = cell.reshape(1, 32, 32, 1)
        prediction = model.predict(cell)[0]
        nombre = np.argmax(prediction)
        print("Prediction : " + str(nombre))


def puzzle(grid):

    grille = []

    #parcourt toutes les cellules
    for i in range(0, 9):
        #reset de tab
        tab = []
        for j in range(0, 9):

            nombre = grid[i][j]
            tab.append(nombre)

        grille.append(tab)

    return grille


def solve(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
             
    for x in range(9):
        if grid[x][col] == num:
            return False
 
 
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True
 
def Sudoku(grid, row, col):
 
    if (row == M - 1 and col == M):
        return True
    if col == M:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Sudoku(grid, row, col + 1)
    for num in range(1, M + 1, 1): 
     
        if solve(grid, row, col, num):
         
            grid[row][col] = num
            if Sudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False