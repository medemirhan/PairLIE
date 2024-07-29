import torch
import torch.nn as nn
from torch.nn import init

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
    def __init__(self, inp_size=32, num=64):
        super(L_net, self).__init__()
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inp_size, num, 3, 1, 0),
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
    def __init__(self, inp_size=32, num=64):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inp_size, num, 3, 1, 0),
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
            nn.Conv2d(num, inp_size, 3, 1, 0),
        )

    def forward(self, input):
        return torch.relu(self.R_net(input))

class N_net(nn.Module):
    def __init__(self, inp_size=32, num=64):
        super(N_net, self).__init__()
        self.N_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inp_size, num, 3, 1, 0),
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
            nn.Conv2d(num, inp_size, 3, 1, 0),
        )

    def forward(self, input):
        return torch.relu(self.N_net(input))


class SpectralConv(nn.Module):
    def __init__(self, num_filters):
        super(SpectralConv, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, num_filters, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(1, num_filters, kernel_size=5, padding=2)
        self.conv3d_3 = nn.Conv3d(1, num_filters, kernel_size=7, padding=3)

    def forward(self, spectral_volume):
        spectral_volume = spectral_volume.unsqueeze(1)  # Shape: (batch_size, 1, bands, height, width)
        conv1 = torch.relu(self.conv3d_1(spectral_volume))  # Shape: (batch_size, num_filters, bands, height, width)
        conv2 = torch.relu(self.conv3d_2(spectral_volume))  # Shape: (batch_size, num_filters, bands, height, width)
        conv3 = torch.relu(self.conv3d_3(spectral_volume))  # Shape: (batch_size, num_filters, bands, height, width)
        output_volume = torch.cat([conv1, conv2, conv3], dim=1)
        return output_volume

class ConvolutionBlock(nn.Module):
    def __init__(self, num_filters, output_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_6 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_7 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_8 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2d_9 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(num_filters * 4, 1, kernel_size=3, padding=1)

    def forward(self, volume):
        conv1 = torch.relu(self.conv2d_1(volume))
        conv2 = torch.relu(self.conv2d_2(conv1))
        conv3 = torch.relu(self.conv2d_3(conv2))
        conv4 = torch.relu(self.conv2d_4(conv3))
        conv5 = torch.relu(self.conv2d_5(conv4))
        conv6 = torch.relu(self.conv2d_6(conv5))
        conv7 = torch.relu(self.conv2d_7(conv6))
        conv8 = torch.relu(self.conv2d_8(conv7))
        conv9 = torch.relu(self.conv2d_9(conv8))
        final_volume = torch.relu(torch.cat([conv3, conv5, conv7, conv9], dim=1))
        clean_band = torch.relu(self.final_conv(final_volume))
        return clean_band

class Network(nn.Module):
    def __init__(self, num_3d_filters, num_conv_filters, input_channels):
        super(Network, self).__init__()
        self.spectral_conv = SpectralConv(num_3d_filters)
        self.convolution_block = ConvolutionBlock(num_3d_filters * 3, num_conv_filters)

    def forward(self, spectral_volume):
        spectral_vol = self.spectral_conv(spectral_volume)
        # Reshape to merge the bands dimension with the batch dimension
        batch_size, num_filters, bands, height, width = spectral_vol.shape
        spectral_vol = spectral_vol.permute(0, 2, 1, 3, 4).reshape(batch_size * bands, num_filters, height, width)
        residue = self.convolution_block(spectral_vol)
        # Reshape back to the original batch size
        residue = residue.view(batch_size, bands, -1, height, width).permute(0, 2, 1, 3, 4).reshape(batch_size, -1, height, width)
        return residue  # Output shape will be the same as the input spatial dimensions (height, width)

class net(nn.Module):
    def __init__(self, num_3d_filters, num_conv_filters, input_channels):
        super(net, self).__init__()        
        self.L_net = L_net(inp_size=input_channels, num=64)
        self.R_net = Network(num_3d_filters, num_conv_filters, input_channels)
        self.N_net = Network(num_3d_filters, num_conv_filters, input_channels)

    def forward(self, input):
        x = self.N_net(input)
        L = self.L_net(x)
        R = self.R_net(x)
        return L, R, x

class net3(nn.Module):
    def __init__(self, inp_size=32):
        super(net3, self).__init__()        
        self.L_net = L_net(inp_size=inp_size, num=64)
        self.R_net = R_net(inp_size=inp_size, num=64)
        self.N_net = N_net(inp_size=inp_size, num=64)

    def forward(self, input):
        x = self.N_net(input)
        L = self.L_net(x)
        R = self.R_net(x)
        return L, R, x
