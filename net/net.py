import torch
import torch.nn as nn

class Fea_net_1_out(nn.Module):
    def __init__(self):
        super(Fea_net_1_out, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 3, 3), stride=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(1, 3, 3), stride=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1)

        self.deconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=(1, 3, 3), stride=1)
        self.bn4 = nn.BatchNorm3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool1(x)

        x = self.deconv1(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.deconv2(x)
        x = self.bn4(x)

        return x

class Fea_net_64_out(nn.Module):
    def __init__(self):
        super(Fea_net_64_out, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 3, 3), stride=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(1, 3, 3), stride=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1)

        self.deconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=1)
        self.bn4 = nn.BatchNorm3d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool1(x)

        x = self.deconv1(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.deconv2(x)
        x = self.bn4(x)

        return x

class L_net(nn.Module):
    def __init__(self, num=64):
        super(L_net, self).__init__()
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_net(input))


class R_net(nn.Module):
    def __init__(self, num=64):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 64, 3, 1, 0),
        )

    def forward(self, input):
        return torch.relu(self.R_net(input))

class N_net(nn.Module):
    def __init__(self, num=64):
        super(N_net, self).__init__()
        self.N_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 64, 3, 1, 0),
        )

    def forward(self, input):
        return torch.relu(self.N_net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        
        '''self.L_net = L_net(num=256)
        self.R_net = R_net(num=256)
        self.N_net = N_net(num=256)'''
        self.L_net = Fea_net_1_out()
        self.R_net = Fea_net_64_out()
        self.N_net = Fea_net_64_out()

    def forward(self, input):
        x = self.N_net(input)
        L = self.L_net(x)
        R = self.R_net(x)
        return L, R, x
