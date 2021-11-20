import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#class ModelBuilder:



#préparation des images
def preProccess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

def build():
    chemin = "myData"
    listeDossiers = os.listdir(chemin)
    nbClasses = len(listeDossiers)
    images = []
    classNumber = []

    testRatio = 0.2
    validationRatio = 0.2
    imageDimensions = (32, 32, 3)

    for x in range(0, nbClasses):
        imageList = os.listdir(chemin +'/'+ str(x))
        for y in imageList:
            curImage = cv2.imread(chemin +'/'+ str(x)+"/"+ y)
            curImage = cv2.resize(curImage, (imageDimensions[0], imageDimensions[1]))
            images.append(curImage)
            classNumber.append(x)

    images = np.array(images)
    classNumber = np.array(classNumber)

    print(images.shape)
    #print(classNumber.shape)

    ### spliting the data
    x_train, x_test, y_train, y_test = train_test_split(images, classNumber, test_size=testRatio)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationRatio)
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)


    numOfSamples = []
    for x in range(0, nbClasses):
        numOfSamples.append(len(np.where(y_train==x)[0]))



    x_train = np.array(list(map(preProccess, x_train)))
    x_test = np.array(list(map(preProccess, x_test)))
    x_validation = np.array(list(map(preProccess, x_validation)))

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)


    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)


    dataGenerator = ImageDataGenerator(width_shift_range=0.1, 
                                        height_shift_range=0.1,
                                        zoom_range=0.1,
                                        shear_range=0.1,
                                        rotation_range=10)


    dataGenerator.fit(x_train)

    y_train = to_categorical(y_train, nbClasses)
    y_test = to_categorical(y_test, nbClasses)
    y_validation = to_categorical(y_validation, nbClasses)


    nbOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    nbOfNodes = 500

    model = Sequential()
    model.add((Conv2D(nbOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(nbOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(nbOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(nbOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nbOfNodes, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nbClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    batchSizeVal = 50
    epochsVal = 5


    model.fit(dataGenerator.flow(x_train, y_train, batch_size=batchSizeVal), 
                                epochs=epochsVal, 
                                validation_data=(x_validation, y_validation),
                                shuffle=1)
    
    model.save('model_train.h5')
    return model

""" 
def showResults(self):


    #Affichage du nombre d'images de chaque classe

    plt.figure(figsize=(10, 5))
    plt.bar(range(0, nbClasses), numOfSamples)
    plt.title("Number of images for each class")
    plt.xlabel("Class ID")
    plt.ylabel("Number of Images")
    plt.show()

    #Affichage d'une image

    img = preProcessing(x_train[50])
    img = cv2.resize(img, (300, 300))
    cv2.imshow("Preprocess", img)
    cv2.waitKey(0)






    model = buildModel()
    print(model.summary())



    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epochs')

    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')

    #plt.show()
    score = model.evaluate(x_test, y_test, verbose=0)
    print("test score : ", score[0])
    print("test accuracy : ", score[1])

    model.save("model_train.h5")
"""