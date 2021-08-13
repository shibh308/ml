from torch import nn, optim

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.glu = nn.GLU()

        self.conv0 = nn.Conv1d(35, 128, 15)

        self.conv1 = nn.Conv1d(128, 256, 5, 2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 512, 5, 2)
        self.bn2 = nn.BatchNorm1d(512)

        self.r1_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r1_bn1 = nn.BatchNorm1d(512)
        self.r1_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r1_bn2 = nn.BatchNorm1d(512)

        self.r2_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r2_bn1 = nn.BatchNorm1d(512)
        self.r2_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r2_bn2 = nn.BatchNorm1d(512)

        self.r3_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r3_bn1 = nn.BatchNorm1d(512)
        self.r3_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r3_bn2 = nn.BatchNorm1d(512)

        self.r4_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r4_bn1 = nn.BatchNorm1d(512)
        self.r4_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r4_bn2 = nn.BatchNorm1d(512)

        self.r5_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r5_bn1 = nn.BatchNorm1d(512)
        self.r5_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r5_bn2 = nn.BatchNorm1d(512)

        self.r6_conv1 = nn.Conv1d(512, 512, 3, padding='same')
        self.r6_bn1 = nn.BatchNorm1d(512)
        self.r6_conv2 = nn.Conv1d(512, 512, 3, padding='same')
        self.r6_bn2 = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 256, 5)
        self.ps1 = nn.PixelShuffle(2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 128, 5)
        self.ps2 = nn.PixelShuffle(2)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 35, 15)
    

    def forward(self, x):
        x = self.conv0(x)
        x = self.glu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.glu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.glu(x)

        z = x
        y = self.r1_conv1(z)
        y = self.r1_bn1(y)
        y = self.glu(y)
        y = self.r1_conv2(y)
        y = self.r1_bn2(y)
        x = z + y

        z = x
        y = self.r2_conv1(z)
        y = self.r2_bn1(y)
        y = self.glu(y)
        y = self.r2_conv2(y)
        y = self.r2_bn2(y)
        x = z + y

        z = x
        y = self.r3_conv1(z)
        y = self.r3_bn1(y)
        y = self.glu(y)
        y = self.r3_conv2(y)
        y = self.r3_bn2(y)
        x = z + y

        z = x
        y = self.r4_conv1(z)
        y = self.r4_bn1(y)
        y = self.glu(y)
        y = self.r4_conv2(y)
        y = self.r4_bn2(y)
        x = z + y

        z = x
        y = self.r5_conv1(z)
        y = self.r5_bn1(y)
        y = self.glu(y)
        y = self.r5_conv2(y)
        y = self.r5_bn2(y)
        x = z + y

        z = x
        y = self.r6_conv1(z)
        y = self.r6_bn1(y)
        y = self.glu(y)
        y = self.r6_conv2(y)
        y = self.r6_bn2(y)
        x = z + y

        x = self.conv3(x)
        x = self.ps1(x)
        x = self.bn3(x)
        x = self.glu(x)

        x = self.conv4(x)
        x = self.ps2(x)
        x = self.bn4(x)
        x = self.glu(x)

        x = self.conv5(x)

        return x

class Network:
    def __init__(self, device, dlr, dbeta, glr, gbeta):
        # 24, 128, 1
        self.D = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2),
            nn.GLU(),
            nn.Conv2d(128, 256, 3, 2),
            nn.BatchNorm1d(256),
            nn.GLU(),
            nn.Conv2d(256, 512, 3, 2),
            nn.BatchNorm1d(512),
            nn.GLU(),
            nn.Conv2d(512, 1024, 3, 2),
            nn.BatchNorm1d(1024),
            nn.GLU(),
            nn.Flatten(),
            nn.Sigmoid()
        ).to(device)
        self.G = Generator().to(device)
        self.G_optim = optim.Adam(self.G.parameters(), lr=glr, betas=(gbeta, 0.999))
        self.D_optim = optim.Adam(self.D.parameters(), lr=dlr, betas=(dbeta, 0.999))