# from imp import load_module
# from crypt import methods
from flask_sqlalchemy import SQLAlchemy
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
import urllib.request
from flask import Flask, app, render_template, request, redirect, url_for, flash, jsonify,make_response
import keras
import tensorflow as tf
from keras import  models,layers ,datasets
# from keras.utils import to_categorical
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = 'Dental Diagnosis'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=['GET'])
def index():
    return jsonify({'done':'done'})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    new_model = tf.keras.models.load_model('network.h5')
    if request.files:
        print("#####" *10)
        # print(request.files)
        # FileStorage.save()
        file = request.files.get('file')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        dims = new_model.input_shape[1:3]
        image_uri = 'static/uploads/{}'.format(str(filename))
        im = keras.preprocessing.image.load_img(image_uri, target_size=dims)
        doc = keras.preprocessing.image.img_to_array(im) # -> numpy arra
        #print(type(doc), doc.shape)
        doc = np.expand_dims(doc, axis=0)
        predictions = new_model.predict(doc)
        classes_names=['noncancer','cancer']
        classes_x=np.argmax(predictions,axis=1)
        print("#####" *30)
        res = np.array(classes_names)[classes_x]
        print(res)
        return jsonify({'res':res[0]})
        



if __name__ == '__main__':
    app.run(debug=True,port=5000)



