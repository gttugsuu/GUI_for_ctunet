import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

class cascadable_TUNET(nn.Module):
    def __init__(self, n_channels, o_channels=3, use_sigmoid=False, fc_bias=True):
        super(cascadable_TUNET, self).__init__()
        self.n_channels = n_channels
        self.o_channels = o_channels
        self.use_sigmoid = use_sigmoid
        self.fc_bias = fc_bias

        self.conv11 = nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2)

        self.conv51 = nn.Conv2d(512, 1023, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.drop5 = nn.Dropout(0.5)

        self.up61 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv61 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv62 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up71 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv71 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv72 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up81 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv81 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv82 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up91 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv91 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv92 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(64, o_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()

        self.tuning_net = torch.nn.Sequential(
            torch.nn.Linear(2,4, bias=self.fc_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(4,16, bias=self.fc_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(16,64, bias=self.fc_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(64,256, bias=self.fc_bias),
            torch.nn.ReLU(),
            )

    def forward(self, x, p, skipc1=None, skipc2=None, skipc3=None):

        tuning = self.tuning_net(torch.cat([p,p], dim=1)).reshape(-1, 1, 16, 16)
      
        conv1 = self.relu(self.conv11(x))
        if skipc1 is None:
            skipc1 = torch.zeros_like(conv1)
        conv1 = torch.cat([skipc1, conv1], 1)
        conv1 = self.relu(self.conv12(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv21(pool1))
        if skipc2 is None:
            skipc2 = torch.zeros_like(conv2)
        conv2 = torch.cat([skipc2, conv2], 1)
        conv2 = self.relu(self.conv22(conv2))
        pool2 = self.pool2(conv2)

        conv3 = self.relu(self.conv31(pool2))
        if skipc3 is None:
            skipc3 = torch.zeros_like(conv3)
        conv3 = torch.cat([skipc3, conv3], 1)
        conv3 = self.relu(self.conv32(conv3))
        pool3 = self.pool3(conv3)

        conv4 = self.relu(self.conv42(self.relu(self.conv41(pool3))))
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        # upsamplings        
        conv51 = self.relu(self.conv51(pool4))
        concat = torch.cat((tuning, conv51), 1)
        conv5 = self.relu(self.conv52(concat))
        drop5 = self.drop5(conv5)

        up6 = self.up61(drop5)
        concat = torch.cat((drop4, up6), 1)
        conv6 = self.relu(self.conv62(self.relu(self.conv61(concat))))

        up7 = self.up71(conv6)
        concat = torch.cat((conv3, up7), 1)
        conv7 = self.relu(self.conv72(self.relu(self.conv71(concat))))

        up8 = self.up81(conv7)
        concat = torch.cat((conv2, up8), 1)
        conv8 = self.relu(self.conv82(self.relu(self.conv81(concat))))

        up9 = self.up91(conv8)
        concat = torch.cat((conv1, up9), 1)
        conv9 = self.relu(self.conv92(self.relu(self.conv91(concat))))

        conv10 = self.conv10(conv9)
        if self.use_sigmoid: conv10 = self.sigmoid(conv10)

        return conv10, conv1, conv2, conv3
