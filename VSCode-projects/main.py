import findsudokugrid
import cv2
import numpy as np

#load the image
image = cv2.imread('sudokuboard.png')

#will create a top down view of the grid
transformed = findsudokugrid.getGridContour(image)

#will create all the lines of the grid
transformed = findsudokugrid.getGridLines(transformed)

cellLocation = findsudokugrid.findCellLocation(transformed, 1, 1)
print(cellLocation)
#cell = findsudokugrid.getTopDownView(transformed, cellLocation)

'''
#loop over all the cells in the grid
for i in range(0, 8):
    for j in range(0, 8):
        #will get location of a specific cell
        cellLocation = findsudokugrid.findCellLocation(transformed, j, i)
'''


#show the image
cv2.imshow('sudokuboard', transformed)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()