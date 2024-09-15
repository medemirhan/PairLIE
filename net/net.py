import torch
import torch.nn as nn
from torch.nn import init

class L_net(nn.Module):
    def __init__(self, inp_size=3, num=64):
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
    def __init__(self, inp_size=3, num=64):
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
        return torch.sigmoid(self.R_net(input))

class N_net(nn.Module):
    def __init__(self, inp_size=3, num=64):
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
        return torch.sigmoid(self.N_net(input))


#3D spectral conv
class SpectralConv(nn.Module):
    def __init__(self, in_channel, out_channel, K=24):
        super(SpectralConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv3d_3 = nn.Conv3d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(K, 3, 3), padding=(0, 1, 1))
        self.conv3d_5 = nn.Conv3d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(K, 5, 5), padding=(0, 2, 2))
        self.conv3d_7 = nn.Conv3d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(K, 7, 7), padding=(0, 3, 3))
        self.relu = nn.ReLU()

    def forward(self, spectral_vol):
        conv1 = torch.squeeze(self.conv3d_3(spectral_vol), dim=2)
        conv2 = torch.squeeze(self.conv3d_5(spectral_vol), dim=2)
        conv3 = torch.squeeze(self.conv3d_7(spectral_vol), dim=2)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)
        output = self.relu(concat_volume)
        return output

#2D spatial conv
class SpatialConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpatialConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv2d_3 = nn.Conv2d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d_5 = nn.Conv2d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(5, 5), padding=(2, 2))
        self.conv2d_7 = nn.Conv2d(in_channels=in_channel, out_channels=
                                  out_channel, kernel_size=(7, 7), padding=(3, 3))
        self.relu = nn.ReLU()

    def forward(self, spatial_band):
        conv1 = self.conv2d_3(spatial_band)
        conv2 = self.conv2d_5(spatial_band)
        conv3 = self.conv2d_7(spatial_band)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)
        output = self.relu(concat_volume)
        return output

class R_net2(nn.Module):
    def __init__(self, spectral_in_channels, spectral_out_channels, spatial_out_channels, K=24):
        super(R_net2, self).__init__()
        self.spectral_conv = SpectralConv(spectral_in_channels, spectral_out_channels, K)
        self.spatial_conv = SpatialConv(spatial_out_channels * 3, spatial_out_channels)
        
    def forward(self, x):
        # Input x shape: (batch_size, in_channel, K, height, width)
        spectral_output = self.spectral_conv(x)
        
        # spectral_output shape: (batch_size, spectral_out_channels * 3, height, width)
        spatial_output = self.spatial_conv(spectral_output)
        
        # spatial_output shape: (batch_size, spatial_out_channels * 3, height, width)
        # Adjusting to match the input size (batch_size, inp_size, height, width)
        output = spatial_output[:, :x.size(1), :, :]  # Crop to match the original input channel size
        
        return output

        # Example usage:
        # spectral_in_channels = inp_size, spectral_out_channels = some value, K = some value
        # spatial_out_channels = inp_size

class L_net2(nn.Module):
    def __init__(self, spectral_in_channels, spectral_out_channels, spatial_out_channels, K=24):
        super(L_net2, self).__init__()
        self.spectral_conv = SpectralConv(spectral_in_channels, spectral_out_channels, K)
        self.spatial_conv = SpatialConv(spatial_out_channels * 3, spatial_out_channels)
        self.final_conv = nn.Conv2d(spatial_out_channels * 3, 1, kernel_size=(1, 1))  # 1x1 Conv to reduce channels to 1
        
    def forward(self, x):
        # Input x shape: (batch_size, in_channel, K, height, width)
        spectral_output = self.spectral_conv(x)
        
        # spectral_output shape: (batch_size, spectral_out_channels * 3, height, width)
        spatial_output = self.spatial_conv(spectral_output)
        
        # spatial_output shape: (batch_size, spatial_out_channels * 3, height, width)
        output = self.final_conv(spatial_output)
        
        # output shape: (batch_size, 1, height, width)
        return output

        # Example usage:
        # spectral_in_channels = inp_size, spectral_out_channels = some value, K = some value
        # spatial_out_channels = some value

class net(nn.Module):
    def __init__(self, inp_size=3):
        super(net, self).__init__()        
        self.L_net = L_net(inp_size=inp_size, num=64) # inp: (batch_size, inp_size, height, width), out: (batch_size, 1, height, width)
        self.R_net = R_net(inp_size=inp_size, num=64) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)
        self.N_net = N_net(inp_size=inp_size, num=64) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)

    def forward(self, input):
        x = self.N_net(input) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)
        L = self.L_net(x) # inp: (batch_size, inp_size, height, width), out: (batch_size, 1, height, width)
        R = self.R_net(x) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)
        return L, R, x


class net_new(nn.Module):
    def __init__(self, spectral_in_channels=3, spectral_out_channels=3, spatial_out_channels=3, K=24):
        super(net_new, self).__init__()
        self.L_net2 = L_net2(spectral_in_channels, spectral_out_channels, spatial_out_channels, K) # inp: (batch_size, inp_size, height, width), out: (batch_size, 1, height, width)
        self.R_net2 = R_net2(spectral_in_channels, spectral_out_channels, spatial_out_channels, K) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)
        self.N_net2 = R_net2(spectral_in_channels, spectral_out_channels, spatial_out_channels, K) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)

    def forward(self, input):
        input = input.unsqueeze(2)  # Shape: (batch_size, in_channels, 1, height, width)

        x = self.N_net2(input) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)
        L = self.L_net2(x) # inp: (batch_size, inp_size, height, width), out: (batch_size, 1, height, width)
        R = self.R_net2(x) # inp: (batch_size, inp_size, height, width), out: (batch_size, inp_size, height, width)
        return L, R, x