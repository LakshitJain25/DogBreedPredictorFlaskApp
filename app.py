from flask import Flask, request, render_template
import cv2
import os
import numpy as np
from keras.models import load_model
from PIL import Image

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        file_upload = request.files["file_upload"]
        filename = file_upload.filename
        original_image = Image.open(file_upload)
        original_image.save(os.path.join("static/uploads", "original.jpg"))
        uploaded_image = Image.open(file_upload).resize((224, 224))
        uploaded_image.save(os.path.join("static/uploads", "image.jpg"))
        app.logger.info(uploaded_image)
        model = load_model("dog_breed.h5")
        CLASS_NAMES = ["Scottish Deerhound", "Maltese Dog", "Bernese Mountain Dog"]
        image = cv2.imread(os.path.join("static/uploads/", "image.jpg"))
        image = cv2.resize(image,(224,224))
        image.shape = (1,224,224,3)
        y_pred = model.predict(image)
        prediction = CLASS_NAMES[np.argmax(y_pred)]
        return render_template("index.html",pred=str(prediction),image="../static/uploads/original.jpg")


if __name__ == "__main__":
    app.run(debug=True)
