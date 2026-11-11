"""DIP network used by the original 2D inversion notebooks."""
import torch
from torch import nn

class DoubleConv(nn.Module):
    """Basic U-Net block: two convolutions with ReLU, without BatchNorm, suitable for single-sample fitting."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=True),  # Enable bias.
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling block: MaxPool followed by DoubleConv, doubling the channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Halve the spatial dimensions and double the channels through DoubleConv.
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling block: interpolation, concatenation, and DoubleConv with matching channels."""
    def __init__(self, up_in_channels, low_channels, out_channels, upsample_mode='bilinear'):
        super().__init__()
        # Interpolation upsamples the spatial dimensions without changing the channel count.
        self.up = nn.Upsample(
            scale_factor=2,
            mode=upsample_mode,
            align_corners=True if upsample_mode == 'bilinear' else None
        )
        # Concatenated channels = upsampled feature channels (up_in_channels) + skip feature channels (low_channels).
        self.conv = DoubleConv(up_in_channels + low_channels, out_channels)

    def forward(self, x1, x2):
        # x1: deeper features to upsample, with up_in_channels channels.
        # x2: shallower skip features, with low_channels channels.
        x1 = self.up(x1)  # Match the upsampled spatial dimensions to x2.

        # Correct size differences so x1 and x2 have exactly matching spatial dimensions.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])

        # Concatenate along channels: total channels = up_in_channels + low_channels.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output layer: a 1x1 convolution mapping features to impedance with one output channel."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Dynamic U-Net whose architecture is constructed from a channel list."""
    def __init__(self, n_channels=1, n_classes=1, channel_list=[16, 32, 64]):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channel_list = channel_list  # Channel list: each entry specifies the channel count at that level.
        self.depth = len(channel_list)    # Network depth equals the number of downsampling stages.

        # Build the downsampling path dynamically with a for loop.
        self.downs = nn.ModuleList()
        # Input layer: map n_channels to the first channel count.
        self.inc = DoubleConv(n_channels, channel_list[0])
        # Subsequent downsampling layers map level i channels to level i+1 channels.
        for i in range(self.depth - 1):
            self.downs.append(Down(channel_list[i], channel_list[i+1]))

        # Build the upsampling path dynamically with a for loop.
        self.ups = nn.ModuleList()
        # Upsampling layers map level i channels to level i-1 channels in reverse order.
        for i in range(self.depth - 1, 0, -1):
            # Use the current level channel count as input and the preceding level count as output.
            self.ups.append(Up(channel_list[i] , channel_list[i-1], channel_list[i-1]))

        # Output layer.
        self.outc = OutConv(channel_list[0], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Downsample and store intermediate features.
        features = [self.inc(x)]  # Store the input-layer output.
        for down in self.downs:
            features.append(down(features[-1]))  # Downsample sequentially and store the features.

        # Upsample and concatenate features, starting from the deepest level.
        x = features[-1]  # Deepest features.
        for i in range(len(self.ups)):
            # Match upsampling layers to feature indices in reverse order.
            x = self.ups[i](x, features[-(i + 2)])  # Concatenate the skip features from the preceding level.

        # Process the output.
        logits = self.outc(x)
        return (self.sigmoid(logits) - 0.5) / 0.5