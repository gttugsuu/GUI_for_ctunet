import argparse
import enum
from torch._C import device
from torch.functional import cartesian_prod

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_pil_image

from codes.dataset import drive
from codes.models import cascadable_TUNET

class test_ctunet:
    def __init__(self, types=['cannyedge', 'binary', 'dilation', 'erosion'], cudas=[0,0,0,0]):
        self.types = types
        self.cudas = cudas

        self.dataset = drive(mode='train', img_type='L')

        self.devices = [torch.device(f'cuda:{cuda}') for cuda in self.cudas]
        print('Using', self.devices)

        self.models = []
        for i, type in enumerate(self.types):
            model = cascadable_TUNET(n_channels=1, o_channels=1).to(self.devices[i])
            model.load_state_dict(torch.load(f'codes/model_weights/ctunet_modules_gray/{type}.pth'))
            model.eval()
            self.models.append(model)
        
        self.maeloss = nn.L1Loss()

    def run_test(self, image_path, pns):
        self.image = self.dataset.transform(Image.open(image_path).convert('L'))
        self.image = self.image.unsqueeze(0).to(self.devices[0])

        ps = []
        for i, pn in enumerate(pns):
            ps.append(torch.tensor([pn]).unsqueeze(-1).float().to(self.devices[i]))

        for m_n, model in enumerate(self.models):
            if m_n == 0:
                output, conv1, conv2, conv3 = model(self.image, ps[0])
            else:
                output = output.to(self.devices[m_n])
                conv1 = conv1.to(self.devices[m_n])
                conv2 = conv2.to(self.devices[m_n])
                conv3 = conv3.to(self.devices[m_n])
                output, conv1, conv2, conv3 = model(output, p=ps[m_n], skipc1=conv1, skipc2=conv2, skipc3=conv3)
        
        output = self.dataset.__detransform__(output.squeeze(0).detach().cpu())
        output = to_pil_image(output)

        return output
