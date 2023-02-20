import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):

  def __init__(self, class_num=2, architecture="resnet18"):
    super(ResNet,self).__init__()
    model = models.resnet18()
    fc_input_dim = model.fc.in_features
    model.fc = nn.Linear(fc_input_dim, class_num)
    self.model = model
    
  def forward(self, x):
    x = self.model(x)
    return x
