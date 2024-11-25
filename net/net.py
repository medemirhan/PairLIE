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

# Adjusted SpectralTransformer Module
class SpectralTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=3, dropout=0.1):
        super(SpectralTransformer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # Residual connection
        x_norm = self.layer_norm2(x)
        x = x + self.feed_forward(x_norm)  # Residual connection
        return x

# Adjusted SpatialTransformer Module
class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):  # Since sequence length is 1
        super(SpatialTransformer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # Residual connection
        x_norm = self.layer_norm2(x)
        x = x + self.feed_forward(x_norm)  # Residual connection
        return x

# SpectralConv remains the same
class SpectralConv(nn.Module):
    def __init__(self, in_channel=1, out_channels=64):
        super(SpectralConv, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channels, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(in_channel, out_channels, kernel_size=5, padding=2)
        self.conv3d_3 = nn.Conv3d(in_channel, out_channels, kernel_size=7, padding=3)
        self.relu = nn.GELU()

    def forward(self, spectral_volume):
        spectral_volume = spectral_volume.unsqueeze(1)  # (batch_size, 1, bands, height, width)
        conv1 = self.conv3d_1(spectral_volume)
        conv2 = self.conv3d_2(spectral_volume)
        conv3 = self.conv3d_3(spectral_volume)
        concat_volume = torch.cat([conv1, conv2, conv3], dim=1)  # (batch_size, out_channels*3, bands, height, width)
        output_volume = self.relu(concat_volume)
        return output_volume

# SpatialConv remains the same
class SpatialConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpatialConv, self).__init__()
        self.conv2d_3 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2d_5 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2)
        self.conv2d_7 = nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3)
        self.relu = nn.GELU()

    def forward(self, spatial_band):
        conv1 = self.conv2d_3(spatial_band)
        conv2 = self.conv2d_5(spatial_band)
        conv3 = self.conv2d_7(spatial_band)
        concat_volume = torch.cat([conv1, conv2, conv3], dim=1)  # (batch_size, out_channel*3, height, width)
        output = self.relu(concat_volume)
        return output

# ConvBlock remains the same
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x):
        return self.relu(self.conv(x))

# Adjusted SpectralConvNetwork
class SpectralConvNetwork(nn.Module):
    def __init__(self, inp_channels, num_heads=3):
        super(SpectralConvNetwork, self).__init__()
        self.inp_channels = inp_channels
        self.spectral_conv = SpectralConv(in_channel=1, out_channels=inp_channels)
        self.transformer = SpectralTransformer(embed_dim=inp_channels * 3, num_heads=num_heads)
        self.conv2d = nn.Conv2d(inp_channels * 3, inp_channels, kernel_size=3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x):
        # x shape: (batch_size, bands, height, width)
        x = self.spectral_conv(x)  # (batch_size, channels, bands, height, width)
        batch_size, channels, bands, height, width = x.shape
        # Prepare for transformer
        x = x.permute(0, 3, 4, 2, 1)  # (batch_size, height, width, bands, channels)
        x = x.reshape(-1, bands, channels)  # (batch_size * height * width, bands, channels)
        x = x.permute(1, 0, 2)  # (bands, batch_size * height * width, channels)
        x = self.transformer(x)  # Transformer expects (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size * height * width, bands, channels)
        x = x.reshape(batch_size, height, width, bands, channels)
        x = x.permute(0, 4, 3, 1, 2)  # (batch_size, channels, bands, height, width)
        # Reduce bands dimension
        x = x.mean(dim=2)  # (batch_size, channels, height, width)
        x = self.conv2d(x)  # (batch_size, inp_channels, height, width)
        return self.relu(x)

# Adjusted SpatialConvNetwork
class SpatialConvNetwork(nn.Module):
    def __init__(self, inp_channels, num_heads=1):
        super(SpatialConvNetwork, self).__init__()
        self.inp_channels = inp_channels
        self.spatial_conv = SpatialConv(in_channel=inp_channels, out_channel=inp_channels)
        self.transformer = SpatialTransformer(embed_dim=inp_channels * 3, num_heads=num_heads)
        self.conv2d_final = nn.Conv2d(inp_channels * 3, 1, kernel_size=3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.spatial_conv(x)  # (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        # Prepare for transformer
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        x = x.reshape(-1, channels)  # (batch_size * height * width, channels)
        x = x.unsqueeze(1)  # (batch_size * height * width, 1, channels)
        x = x.permute(1, 0, 2)  # (seq_len=1, batch_size * height * width, channels)
        x = self.transformer(x)  # Transformer expects (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size * height * width, seq_len=1, channels)
        x = x.squeeze(1)  # (batch_size * height * width, channels)
        x = x.reshape(batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
        x = self.conv2d_final(x)  # (batch_size, 1, height, width)
        return self.relu(x)

# Main Network with Adjusted Parameters
class net(nn.Module):
    def __init__(self, inp_size=3, num_heads=3):
        super(net, self).__init__()
        self.conv_block = ConvBlock(inp_size, inp_size)
        self.L_net = SpatialConvNetwork(inp_channels=inp_size, num_heads=1)  # Adjusted num_heads
        self.R_net = SpectralConvNetwork(inp_channels=inp_size, num_heads=num_heads)
        self.N_net = SpectralConvNetwork(inp_channels=inp_size, num_heads=num_heads)

    def forward(self, input):
        # input shape: (batch_size, channels, height, width)
        preprocessed_input = self.conv_block(input)  # (batch_size, inp_size, height, width)
        x = self.N_net(preprocessed_input)  # (batch_size, inp_size, height, width)
        L = self.L_net(x)  # (batch_size, 1, height, width)
        R = self.R_net(x)  # (batch_size, inp_size, height, width)
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
