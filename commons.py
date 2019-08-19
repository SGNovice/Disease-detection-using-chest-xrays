import io

import torch 
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image 

import torch.nn.functional as F
def mila(input, beta=-0.25):
    '''
    Applies the Mila function element-wise:
    Mila(x) = x * tanh(softplus(1 + β)) = x * tanh(ln(1 + exp(x+β)))
    See additional documentation for mila class.
    '''
    return input * torch.tanh(F.softplus(input+beta))


class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2208, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 3)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.acivation = mila
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.acivation(self.fc1(x)))
        x = self.dropout(self.acivation(self.fc2(x)))

        x = self.logsoftmax(self.fc3(x))
        return x


def get_model():
    checkpoint_path='xray_projectV4_2_densenet161_mila.pt'
    model=models.densenet161(pretrained=True)
    model.classifier = classifier()
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=False)
    model.eval()
    return model

def get_tensor(image_bytes):
	my_transforms=transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
	image=Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)
