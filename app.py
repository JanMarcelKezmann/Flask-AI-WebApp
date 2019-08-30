from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/your_model.h5'

model = ResNet50(weights="imagenet")

graph = tf.get_default_graph()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route("/videoclassification", methods=["GET"])
def videoclassification():
	return render_template("video.html")

@app.route("/imageclassification", methods=["GET"])
def imageclassfication():
    return render_template("image.html")

@app.route("/predict", methods=["GET", "POST"])
# def imageclassification():
# 	return render_template("image.html")
def upload():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['image']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)

            # Process your result for human
            # pred_class = preds.argmax(axis=-1)            # Simple argmax
            pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
            result = str(pred_class[0][0][1])               # Convert to string
            return result
        return None

@app.route("/movementclassification", methods=["GET"])
def movementclassification():
	return render_template("movement.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
	# app.run(port=5002, debug=True)

	# Serve the app with gevent
	http_server = WSGIServer(('0.0.0.0', 5000), app)
	http_server.serve_forever()

