import torch
import torch.nn as nn
from torch.nn import init

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

# Spectral Attention Module
class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool across height and width
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, channels, bands, height, width)
        y = self.avg_pool(x)  # Shape: (batch_size, channels, bands, 1, 1)
        y = self.conv(y)      # Shape: (batch_size, channels, bands, 1, 1)
        y = self.sigmoid(y)   # Shape: (batch_size, channels, bands, 1, 1)
        return x * y          # Element-wise multiplication

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        avg_out = torch.mean(x, dim=1, keepdim=True)        # Shape: (batch_size, 1, height, width)
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # Shape: (batch_size, 1, height, width)
        y = torch.cat([avg_out, max_out], dim=1)            # Shape: (batch_size, 2, height, width)
        y = self.conv(y)                                    # Shape: (batch_size, 1, height, width)
        y = self.sigmoid(y)                                 # Shape: (batch_size, 1, height, width)
        return x * y                                        # Element-wise multiplication

# Modified SpectralConv with Spectral Attention
class SpectralConv(nn.Module):
    def __init__(self, in_channel=1, out_channels=64):
        super(SpectralConv, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channels, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(in_channel, out_channels, kernel_size=5, padding=2)
        self.conv3d_3 = nn.Conv3d(in_channel, out_channels, kernel_size=7, padding=3)
        self.relu = nn.GELU()
        self.spectral_attention = SpectralAttention(out_channels * 3)  # After concatenation

    def forward(self, spectral_volume):
        spectral_volume = spectral_volume.unsqueeze(1)  # Shape: (batch_size, 1, bands, height, width)
        conv1 = self.conv3d_1(spectral_volume)  # Shape: (batch_size, out_channels, bands, height, width)
        conv2 = self.conv3d_2(spectral_volume)
        conv3 = self.conv3d_3(spectral_volume)
        concat_volume = torch.cat([conv1, conv2, conv3], dim=1)  # Shape: (batch_size, out_channels*3, bands, height, width)
        attention_volume = self.spectral_attention(concat_volume)
        output_volume = self.relu(attention_volume)
        return output_volume

# Modified SpatialConv with Spatial Attention
class SpatialConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpatialConv, self).__init__()
        self.conv2d_3 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2d_5 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2)
        self.conv2d_7 = nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3)
        self.relu = nn.GELU()
        self.spatial_attention = SpatialAttention(kernel_size=7)  # Using a larger kernel size for wider context

    def forward(self, spatial_band):
        conv1 = self.conv2d_3(spatial_band)
        conv2 = self.conv2d_5(spatial_band)
        conv3 = self.conv2d_7(spatial_band)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)  # Shape: (batch_size, out_channel*3, height, width)
        attention_volume = self.spatial_attention(concat_volume)
        output = self.relu(attention_volume)
        return output

# ConvBlock remains the same
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.GELU()
    
    def forward(self, x):
        return self.relu(self.conv(x))

# Modified SpectralConvNetwork with adjusted dimensions
class SpectralConvNetwork(nn.Module):
    def __init__(self, inp_channels):
        super(SpectralConvNetwork, self).__init__()
        self.spectral_conv = SpectralConv(in_channel=1, out_channels=inp_channels)
        self.conv2d = nn.Conv2d(inp_channels * 3, inp_channels, kernel_size=3, padding=1)
        self.relu = nn.GELU()
    
    def forward(self, x):
        x = self.spectral_conv(x)
        # Convert 3D output to 2D by averaging over the spectral dimension
        x = x.mean(dim=2)  # Shape: (batch_size, out_channels*3, height, width)
        x = self.conv2d(x)
        return self.relu(x)

# Modified SpatialConvNetwork with adjusted dimensions
class SpatialConvNetwork(nn.Module):
    def __init__(self, inp_channels):
        super(SpatialConvNetwork, self).__init__()
        self.spatial_conv = SpatialConv(in_channel=inp_channels, out_channel=inp_channels)
        self.conv2d_final = nn.Conv2d(inp_channels * 3, 1, kernel_size=3, padding=1)
        self.relu = nn.GELU()
    
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.conv2d_final(x)
        return self.relu(x)

# Main network remains mostly the same
class net(nn.Module):
    def __init__(self, inp_size=64):
        super(net, self).__init__()
        self.conv_block = ConvBlock(inp_size, inp_size)
        self.L_net = SpatialConvNetwork(inp_channels=inp_size)
        self.R_net = SpectralConvNetwork(inp_channels=inp_size)
        self.N_net = SpectralConvNetwork(inp_channels=inp_size)
    
    def forward(self, input):
        preprocessed_input = self.conv_block(input)
        x = self.N_net(preprocessed_input)
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
