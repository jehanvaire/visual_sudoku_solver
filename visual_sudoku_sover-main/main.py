from genericpath import exists
import findsudokugrid
#import digit_recognizer
import matplotlib.pyplot as plt
#from keras.preprocessing import image
#import tensorflow as tf
import cv2
import numpy as np
import os

#load the image
img = cv2.imread('sudokuboard1.png')

#will create a top down view of the grid
transformed = findsudokugrid.getGridContour(img)

#will create all the lines of the grid
transformed = findsudokugrid.getGridLines(transformed)
cv2.imshow("Grid + lines", transformed)

cellLocation = findsudokugrid.findCellLocation(transformed, 0, 0)
cell = findsudokugrid.getTopDownView(transformed, cellLocation)

#if the image is too big (>28*28) we crop it
if(cell.shape > (28, 28, 3)):
    cell = findsudokugrid.cropTo28(cell)




'''
#create the CNN network if it doesn't exists
if(os.path.exists("number_recognition.h5")):
    model = tf.keras.models.load_model("number_recognition.h5")
else:
    model = digit_recognizer.build()

model.summary()

cell = image.img_to_array(cell)
cell = np.expand_dims(cell, axis=0)
cell = cell.reshape(28, 28, 3)

prediction = model.predict(cell, batch_size=1)
print(prediction)
'''


'''
#loop in all the cells in the grid
for i in range(0, 8):
    for j in range(0, 8):
        #will get location of a specific cell
        cellLocation = findsudokugrid.findCellLocation(transformed, j, i)
'''


#show the image
#cv2.imshow('sudokuboard', cell)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()