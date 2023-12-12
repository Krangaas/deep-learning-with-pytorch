import torch.nn as nn
from torch import flatten

class Resnet18(nn.Module):
    '''
    Implementation of a ResNet-18 deep learning model.
    The model is hardcoded to take inputs with dimension (3,224,224),
    and output 5 class scores ("restriction_signs", "speed_limits", "stop_signs",
                               "warning_signs", "yield_signs").
    '''
    def __init__(self):
        super().__init__()
        ####### INPUT #######
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ######## LAYER 1 #######
        self.l1_rb1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.l1_rb1_bn1 = nn.BatchNorm2d(64)
        self.l1_rb1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.l1_rb1_bn2 = nn.BatchNorm2d(64)

        self.l1_rb2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.l1_rb2_bn1 = nn.BatchNorm2d(64)
        self.l1_rb2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.l1_rb2_bn2 = nn.BatchNorm2d(64)

        ####### LAYER 2 #######
        self.l2_dn_conv = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.l2_dn_bn =  nn.BatchNorm2d(128)

        self.l2_rb1_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.l2_rb1_bn1 = nn.BatchNorm2d(128)
        self.l2_rb1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.l2_rb1_bn2 = nn.BatchNorm2d(128)

        self.l2_rb2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.l2_rb2_bn1 = nn.BatchNorm2d(128)
        self.l2_rb2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.l2_rb2_bn2 = nn.BatchNorm2d(128)

        ####### LAYER 3 #######
        self.l3_dn_conv = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.l3_dn_bn = nn.BatchNorm2d(256)

        self.l3_rb1_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.l3_rb1_bn1 = nn.BatchNorm2d(256)
        self.l3_rb1_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.l3_rb1_bn2 = nn.BatchNorm2d(256)

        self.l3_rb2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.l3_rb2_bn1 = nn.BatchNorm2d(256)
        self.l3_rb2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.l3_rb2_bn2 = nn.BatchNorm2d(256)

        ####### LAYER 4 #######
        self.l4_dn_conv = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.l4_dn_bn = nn.BatchNorm2d(512)

        self.l4_rb1_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.l4_rb1_bn1 = nn.BatchNorm2d(512)
        self.l4_rb1_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.l4_rb1_bn2 = nn.BatchNorm2d(512)

        self.l4_rb2_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.l4_rb2_bn1 = nn.BatchNorm2d(512)
        self.l4_rb2_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.l4_rb2_bn2 = nn.BatchNorm2d(512)

        ####### OUPUT #######
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512, 5, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ######## LAYER 1 #######
        identity = x
        x = self.l1_rb1_conv1(x)
        x = self.l1_rb1_bn1(x)
        x = self.relu(x)
        x = self.l1_rb1_conv2(x)
        x = self.l1_rb1_bn2(x)

        x += identity
        x = self.relu(x)

        identity = x
        x = self.l1_rb2_conv1(x)
        x = self.l1_rb2_bn1(x)
        x = self.relu(x)
        x = self.l1_rb2_conv2(x)
        x = self.l1_rb2_bn2(x)

        x += identity
        x = self.relu(x)

        ####### LAYER 2 #######
        identity = self.l2_dn_conv(x)
        identity = self.l2_dn_bn(identity)
        x = self.l2_rb1_conv1(x)
        x = self.l2_rb1_bn1(x)
        x = self.relu(x)
        x = self.l2_rb1_conv2(x)
        x = self.l2_rb1_bn2(x)

        x += identity
        x = self.relu(x)

        identity = x
        x = self.l2_rb2_conv1(x)
        x = self.l2_rb2_bn1(x)
        x = self.relu(x)
        x = self.l2_rb2_conv2(x)
        x = self.l2_rb2_bn2(x)

        x += identity
        x = self.relu(x)

        ####### LAYER 3 #######
        identity = self.l3_dn_conv(x)
        identity = self.l3_dn_bn(identity)
        x = self.l3_rb1_conv1(x)
        x = self.l3_rb1_bn1(x)
        x = self.relu(x)
        x = self.l3_rb1_conv2(x)
        x = self.l3_rb1_bn2(x)

        x += identity
        x = self.relu(x)

        identity = x
        x = self.l3_rb2_conv1(x)
        x = self.l3_rb2_bn1(x)
        x = self.relu(x)
        x = self.l3_rb2_conv2(x)
        x = self.l3_rb2_bn2(x)

        x += identity
        x = self.relu(x)

        ####### LAYER 4 #######
        identity = self.l4_dn_conv(x)
        identity = self.l4_dn_bn(identity)
        x = self.l4_rb1_conv1(x)
        x = self.l4_rb1_bn1(x)
        x = self.relu(x)
        x = self.l4_rb1_conv2(x)
        x = self.l4_rb1_bn2(x)

        x += identity
        x = self.relu(x)

        identity = x
        x = self.l4_rb2_conv1(x)
        x = self.l4_rb2_bn1(x)
        x = self.relu(x)
        x = self.l4_rb2_conv2(x)
        x = self.l4_rb2_bn2(x)

        x += identity
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.relu(x)
        x = flatten(x, 1)
        x = self.fc(x)
        return x
