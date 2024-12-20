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

class SpectralConv(nn.Module):
    def __init__(self, in_channel=1, out_channels=64):
        super(SpectralConv, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channels, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(in_channel, out_channels, kernel_size=5, padding=2)
        self.conv3d_3 = nn.Conv3d(in_channel, out_channels, kernel_size=7, padding=3)
        self.activation = nn.GELU()

    def forward(self, spectral_volume):
        spectral_volume = spectral_volume.unsqueeze(1)  # Shape: (batch_size, 1, bands, height, width)
        conv1 = self.conv3d_1(spectral_volume)  # Shape: (batch_size, out_channels, bands, height, width)
        conv2 = self.conv3d_2(spectral_volume)  # Shape: (batch_size, out_channels, bands, height, width)
        conv3 = self.conv3d_3(spectral_volume)  # Shape: (batch_size, out_channels, bands, height, width)
        concat_volume = torch.cat([conv1, conv2, conv3], dim=1)
        output_volume = self.activation(concat_volume)
        return output_volume

class SpatialConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpatialConv, self).__init__()
        self.conv2d_3 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d_5 = nn.Conv2d(in_channel, out_channel, kernel_size=(5, 5), padding=(2, 2))
        self.conv2d_7 = nn.Conv2d(in_channel, out_channel, kernel_size=(7, 7), padding=(3, 3))
        self.activation = nn.GELU()

    def forward(self, spatial_band):
        conv1 = self.conv2d_3(spatial_band)
        conv2 = self.conv2d_5(spatial_band)
        conv3 = self.conv2d_7(spatial_band)
        concat_volume = torch.cat([conv3, conv2, conv1], dim=1)
        output = self.activation(concat_volume)
        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_blocks=3):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_blocks = num_conv_blocks
        self.conv_layer = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.final_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x):
        out = x
        for i in range(self.num_conv_blocks - 1):
            out = self.conv_layer(out)
            out = self.activation(out)

        out = self.final_layer(out)
        out = self.activation(out)
        
        return out

# First Network: Uses SpectralConv
class SpectralConvNetwork(nn.Module):
    def __init__(self, inp_channels):
        super(SpectralConvNetwork, self).__init__()
        self.spectral_conv = SpectralConv(in_channel=1, out_channels=inp_channels)
        self.conv2d = nn.Conv2d(inp_channels * 3, inp_channels, kernel_size=3, padding=1)  # Adjust output channels
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.spectral_conv(x)
        # Convert 3D output to 2D
        x = x.mean(dim=2)  # Reduce the 'bands' dimension (assuming it's dim 2)
        x = self.conv2d(x)
        return self.activation(x)

# Second Network: Uses SpatialConv
class SpatialConvNetwork(nn.Module):
    def __init__(self, inp_channels):
        super(SpatialConvNetwork, self).__init__()
        self.spatial_conv = SpatialConv(in_channel=inp_channels, out_channel=inp_channels)
        self.conv2d_final = nn.Conv2d(inp_channels * 3, inp_channels, kernel_size=3, padding=1)  # Reduce to 1 channel
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.conv2d_final(x)
        return self.activation(x)

class FuseNetwork(nn.Module):
    def __init__(self, inp_size=64, fused_out=64, num_conv_blocks=4):
        super(FuseNetwork, self).__init__()
        self.spatial_branch = SpatialConvNetwork(inp_channels=inp_size)
        self.spectral_branch = SpectralConvNetwork(inp_channels=inp_size)
        self.fusion_block = ConvBlock(inp_size * 2, fused_out, num_conv_blocks)
    
    def forward(self, x):
        # x shape: [B, in_channels, H, W]
        # 1. Extract spectral features
        spectral_feats = self.spectral_branch(x)   # [B, in_channels, H, W]
        # 2. Extract spatial features
        spatial_feats = self.spatial_branch(x)     # [B, in_channels, H, W]
        # 3. Fuse them by concatenating along channel dimension
        fused_feats = torch.cat([spectral_feats, spatial_feats], dim=1)  # [B, in_channels+in_channels, H, W]
        # 4. Pass through the fusion block
        output = self.fusion_block(fused_feats)  # [B, fused_out, H, W]

        return output

class net(nn.Module):
    def __init__(self, inp_size=64, num_conv_blocks=4):
        super(net, self).__init__()
        self.L_net = FuseNetwork(inp_size, fused_out=1, num_conv_blocks=num_conv_blocks)
        self.R_net = FuseNetwork(inp_size, fused_out=inp_size, num_conv_blocks=num_conv_blocks)
        self.N_net = FuseNetwork(inp_size, fused_out=inp_size, num_conv_blocks=num_conv_blocks)

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
