from torch import nn, optim


class GLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gate):
        gate_sig = self.sigmoid(gate)
        y = x * gate_sig
        return y


class Conv2dNormGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.glu = GLU()
    
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        z = self.conv_g(x)
        z = self.bn(z)
        x = self.glu(y, z)
        return x


class Conv2dGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.glu = GLU()
    
    def forward(self, x):
        y = self.conv(x)
        z = self.conv_g(x)
        x = self.glu(y, z)
        return x


class PixelShuffle1d(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.glu = GLU()

        self.conv0 = nn.Conv1d(35, 128, 15, padding='same')
        self.conv0_g = nn.Conv1d(35, 128, 15, padding='same')

        self.conv1 = nn.Conv1d(128, 256, 5, 2, padding=2)
        self.conv1_g = nn.Conv1d(128, 256, 5, 2, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 512, 5, 2, padding=2)
        self.conv2_g = nn.Conv1d(256, 512, 5, 2, padding=2)
        self.bn2 = nn.BatchNorm1d(512)

        self.r_bn = nn.BatchNorm1d(512)

        self.r1_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r1_conv1_g = nn.Conv1d(512, 512, 3, padding='same')
        self.r1_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r1_conv2_g = nn.Conv1d(512, 512, 3, padding='same')

        self.r2_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r2_conv1_g = nn.Conv1d(512, 512, 3, padding='same')
        self.r2_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r2_conv2_g = nn.Conv1d(512, 512, 3, padding='same')

        self.r3_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r3_conv1_g = nn.Conv1d(512, 512, 3, padding='same')
        self.r3_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r3_conv2_g = nn.Conv1d(512, 512, 3, padding='same')

        self.r4_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r4_conv1_g = nn.Conv1d(512, 512, 3, padding='same')
        self.r4_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r4_conv2_g = nn.Conv1d(512, 512, 3, padding='same')

        self.r5_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r5_conv1_g = nn.Conv1d(512, 512, 3, padding='same')
        self.r5_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r5_conv2_g = nn.Conv1d(512, 512, 3, padding='same')

        self.r6_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r6_conv1_g = nn.Conv1d(512, 512, 3, padding='same')
        self.r6_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r6_conv2_g = nn.Conv1d(512, 512, 3, padding='same')

        self.conv3 = nn.Conv1d(512, 256, 5, padding='same')
        self.conv3_g = nn.Conv1d(512, 256, 5, padding='same')
        self.ps1 = PixelShuffle1d(2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 64, 5, padding='same')
        self.conv4_g = nn.Conv1d(128, 64, 5, padding='same')
        self.ps2 = PixelShuffle1d(2)
        self.bn4 = nn.BatchNorm1d(32)

        self.conv5 = nn.Conv1d(32, 35, 15, padding='same')
    

    def forward(self, x):
        y = self.conv0(x)
        z = self.conv0_g(x)
        x = self.glu(y, z)

        y = self.conv1(x)
        z = self.conv1_g(x)
        y = self.bn1(y)
        z = self.bn1(z)
        x = self.glu(y, z)

        y = self.conv2(x)
        z = self.conv2_g(x)
        y = self.bn2(z)
        z = self.bn2(z)
        x = self.glu(y, z)

        i = x
        y = self.r1_conv1(x)
        z = self.r1_conv1_g(x)
        y = self.r_bn(y)
        z = self.r_bn(z)
        o = self.glu(y, z)
        x = i + o

        y = self.conv3(x)
        y = self.ps1(y)
        y = self.bn3(y)
        z = self.conv3_g(x)
        z = self.ps1(z)
        z = self.bn3(z)
        x = self.glu(y, z)

        y = self.conv4(x)
        y = self.ps2(y)
        y = self.bn4(y)
        z = self.conv4_g(x)
        z = self.ps2(z)
        z = self.bn4(z)
        x = self.glu(y, z)

        x = self.conv5(x)

        return x

class Network:
    def __init__(self, device, dlr, dbeta, glr, gbeta):
        # 24, 128, 1
        # def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        self.D = nn.Sequential(
            Conv2dGLU(1, 128, 3, 2, 1),
            Conv2dNormGLU(128, 256, 3, 2, 1),
            Conv2dNormGLU(256, 512, 3, 2, 1),
            Conv2dNormGLU(512, 1024, 3, 2, 1),
            Conv2dNormGLU(1024, 1024, (1, 5), 1),
            nn.Conv2d(1024, 1, (3, 4), 1),
            nn.Flatten(),
            nn.Sigmoid()
        ).to(device)
        self.G = Generator().to(device)
        self.G_optim = optim.Adam(self.G.parameters(), lr=glr, betas=(gbeta, 0.999))
        self.D_optim = optim.Adam(self.D.parameters(), lr=dlr, betas=(dbeta, 0.999))