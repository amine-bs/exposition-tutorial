from io import BytesIO
from flask import Flask, render_template, request
from utils import load_device, import_model, predict, load_image
from PIL import Image
import os

def read_image(file):
    img = Image.open(BytesIO(file))
    return img

app = Flask(__name__)

device = load_device()
model = import_model(bucket="mbenxsalha", key="diffusion/state_dict.pickle", device=device)


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('home.html')

@app.route("/predict", methods=["GET", "POST"])
def predict_flask():
    if request.method == "POST":
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static', filename)
        file.save(file_path)
        img = load_image(file_path)
        pred = predict(img, model, device)
    return render_template("predict.html", output=pred, user_image = file_path)
