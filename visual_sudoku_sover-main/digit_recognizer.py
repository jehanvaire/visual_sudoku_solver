from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.backend import dropout
from sklearn.metrics import classification_report


def build():
    #initialize the model
    model = Sequential()
    inputShape = (28, 28, 1)

    #first set of conv > relu > pool layers
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #second set of conv > relu > pool layers
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #first set of fc > relu layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    #second set of fc > relu layers
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    #softmax classifier
    model.add(Dense(10))
    model.add(Activation("softmax"))


    #initialize the learning rate, epochs, batch size
    init_lr = 1e-3
    epochs = 10
    bs = 128

    #load the mnist dataset
    print("[+] Loading dataset")
    ((trainData, trainLabel), (testData, testLabel)) = mnist.load_data()

    #add a channel (grayscale) dimension to the digits
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    #scale data to the range of [0, 1]
    trainData = trainData.astype("float32")/255.0
    testData = testData.astype("float32")/255.0

    #convert the labels from integer to vectors
    le = LabelBinarizer()
    trainLabel = le.fit_transform(trainLabel)
    testLabel = le.fit_transform(testLabel)

    #initialize the optimizer
    print("[+] compiling model...")
    opt = Adam(learning_rate=init_lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    #train the model
    print("[+] Training network...")
    model.fit(trainData, trainLabel, 
        validation_data=(testData, testLabel), 
        batch_size=bs, epochs=epochs, verbose=1)

    #evaluate the model
    print("[+] evaluating network...")
    predictions = model.predict(testData)
    print(classification_report(testLabel.argmax(axis=1), 
        predictions.argmax(axis=1), target_names=[str(x) 
        for x in le.classes_]))

    #save the model
    print("[+] saving model")
    model.save("number_recognition.h5")