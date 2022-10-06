from model import ResNet
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import streamlit as st
import boto3
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

@st.cache()
def import_model(bucket, key="diffusion/state_dict.pickle", device=torch.device('cpu')):
    s3 = boto3.client('s3',endpoint_url='https://minio.lab.sspcloud.fr/')
    data = s3.get_object(Bucket=bucket, Key=key)
    state_dict = pickle.loads(data['Body'].read())
    model = ResNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_image(file):
    img = Image.open(file).convert("RGB")
    return img

@st.cache()
def predict(img, model, device):
    img = TF.to_tensor(img)
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.to(device)
    img = img.unsqueeze(0)
    preds = model(img)
    preds = F.softmax(preds, dim=1)
    if float(preds[0][0]) < float(preds[0][1]):
        results = "It is a Pizza!"
    else:
        results = "It is not a Pizza!"
    return results

@st.cache()
def load_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

@st.cache()
def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)