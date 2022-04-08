from flask import Flask, request
from werkzeug.utils import secure_filename
from main import main
import os
import digit_recognizer
import tensorflow as tf



# INITIALISATION VARIABLES #
adresse = '192.168.1.55'
port = '5000'
debug = False
# FIN INITIALISATION VARIABLES #



app = Flask(__name__)



@app.route('/image/create', methods=['POST'])
def upload():

    imageFile = request.files['image']
    filename = secure_filename(imageFile.filename)
    imageFile.save(filename)
    prediction = main(imageFile.filename, model)
    # remove the file
    os.remove(imageFile.filename)

    prediction = str(prediction)

    return prediction




if __name__ == '__main__':
    # #create the CNN network if it doesn't exists
    if(os.path.exists("model_train.h5")):
        model = tf.keras.models.load_model("model_train.h5")
    else:
        model = digit_recognizer.build()

    
    app.run(host=adresse, port=port, debug=debug)

