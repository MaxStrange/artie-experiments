from torch import nn
import torch

class ResidualBlock(nn.Module):
    """
    Classic Residual block. Stolen shamelessly from
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
    under MIT license.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.lrelu(out)
        return out

class ResNet(nn.Module):
    """
    Simple ResNet stolen shamelessly from
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
    under MIT license (then adjusted for our use case).
    """
    def __init__(self, ninput_channels, reslayer_kernels=(32, 64)):
        """
        reslayer_kernels is the number of kernels for each res layer after the first
        """
        super().__init__()
        self.in_channels = 16  # Starting number of feature maps
        self.conv = nn.Conv2d(ninput_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)

        reslayers = [self.make_res_block(self.in_channels, 2)]
        for nkernels in reslayer_kernels:
            layer = self.make_res_block(nkernels, 2, 2)
            reslayers.append(layer)
        self.reslayers = nn.Sequential(*reslayers)

        self.avg_pool = nn.AvgPool2d(12)
        self.flatten = nn.Flatten()
        
    def make_res_block(self, out_channels, nblocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # First block may need to downsample
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        for _ in range(1, nblocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        self.in_channels = out_channels  # Update feature map dimension
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        for layer in self.reslayers:
            out = layer(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, upscale_factor: int, inchannels: int, outchannels: int, batchnorm=True) -> None:
        super().__init__()
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        shuffledchannels = int(inchannels / (upscale_factor * upscale_factor))
        self.conv2d = nn.Conv2d(shuffledchannels, outchannels, (3, 3), padding=(1, 1))
        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(outchannels)
        else:
            self.batchnorm = None
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.pixelshuffle(x)
        out = self.conv2d(out)
        if self.batchnorm:
            out = self.batchnorm(out)
        out = self.lrelu(out)
        return out

class InterpolateBlock(nn.Module):
    def __init__(self, scale: int, inchannels: int, outchannels: int) -> None:
        super().__init__()
        self.shuffle = nn.PixelShuffle(scale)
        needed_channels = int(outchannels * scale * scale)
        self.nrepeats = max(int(needed_channels / inchannels), 1)

    def forward(self, x):
        out = torch.repeat_interleave(x, self.nrepeats, 1)
        out = self.shuffle(out)
        return out

class FancyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # First branch: upsampling to learn residual using trained params
        self.upblock4x4 = UpsampleBlock(4, 512, 512, batchnorm=False)
        self.upblock8x8 = UpsampleBlock(2, 512, 256)
        self.upblock16x16 = UpsampleBlock(2, 256, 128, batchnorm=False)
        self.upblock32x32 = UpsampleBlock(2, 128, 64)
        self.convblock25x28 = nn.Sequential(
            nn.Conv2d(64, 64, (4, 3)),     # -> 29x30
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (3, 2)),     # -> 27x29
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (3, 2)),     # -> 25x28
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.upblock50x56 = UpsampleBlock(2, 64, 32)
        self.upblock100x112 = UpsampleBlock(2, 32, 1, batchnorm=False)

        # Second branch: upsampling using non-learned params
        self.upsample8x8 = InterpolateBlock(8, 512, 256)
        self.upsample16x16 = InterpolateBlock(2, 256, 128)
        self.upsample32x32 = InterpolateBlock(2, 128, 64)
        self.upsample50x56 = InterpolateBlock(2, 64, 32)
        self.upsample100x112 = InterpolateBlock(2, 32, 1)

        # Braches meet up and combine the residual, then some fiddling to get the exact dims needed
        self.tconv101x113 = nn.ConvTranspose2d(1, 1, (2, 2))
        self.upsample202x113 = nn.Upsample(scale_factor=(2, 1))

    def forward(self, x):
        out = self.upblock4x4(x)

        out = self.upblock8x8(out)
        x = self.upsample8x8(x)
        out = out + x

        out = self.upblock16x16(out)
        x = self.upsample16x16(x)
        out = out + x

        out = self.upblock32x32(out)
        x = self.upsample32x32(x)
        out = out + x

        # Convolve down
        out = self.convblock25x28(out)
        x = torch.nn.functional.interpolate(x, size=(25, 28))

        out = self.upblock50x56(out)
        x = self.upsample50x56(x)
        out = out + x

        out = self.upblock100x112(out)
        x = self.upsample100x112(x)
        x = torch.mean(x, 1, keepdim=True)
        out = out + x
        
        # Merged branch
        out = self.tconv101x113(out)
        out = self.upsample202x113(out)
        out = out[:, :, 1:, :]  # Strip off one row of pixels as the network outputs 1 too many rows
        return out