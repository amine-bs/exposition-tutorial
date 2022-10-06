import torch
from utils import *

device = load_device()
model = import_model(bucket="mbenxsalha", key="diffusion/state_dict.pickle", device=device)

st.title("Welcome to The Ship Detective!")
st.write("The image you upload will be fed to a Deep Neural Network in real-time to verify if there is a ship or not")
file = st.file_uploader("Upload an image")

if file:
    img = load_image(file)
    predictions = predict(img, model, device)
    st.title("Here is the image you uploaded")
    resized_image = img.resize((340, 340))
    st.image(resized_image)
    st.title("Prediction:")
    st.write(predictions)