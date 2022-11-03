from torch import nn

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
    def __init__(self, ninput_channels, num_out, reslayer_kernels=(32, 64)):
        """
        reslayer_kernels is the number of kernels for each res layer after the first
        """
        super().__init__()
        self.in_channels = 16  # Starting number of feature maps
        self.conv = nn.Conv2d(ninput_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.reslayers = [self.make_res_block(self.nchannels, 2)]
        for nkernels in reslayer_kernels:
            layer = self.make_res_block(nkernels, 2, 2)
            self.reslayers.append(layer)

        self.avg_pool = nn.AvgPool2d(12)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_out)
        
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
        out = self.fc(out)
        return out

