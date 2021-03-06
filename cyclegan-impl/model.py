from torch import nn, optim

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.r1_conv1 = nn.Conv2d(128, 128, 1, bias=False)
        self.r1_bn1 = nn.BatchNorm2d(128)
        self.r1_conv2 = nn.Conv2d(128, 128, 1, bias=False)
        self.r1_bn2 = nn.BatchNorm2d(128)

        """
        self.r2_conv1 = nn.Conv2d(128, 128, 1, bias=False)
        self.r2_bn1 = nn.BatchNorm2d(128)
        self.r2_conv2 = nn.Conv2d(128, 128, 1, bias=False)
        self.r2_bn2 = nn.BatchNorm2d(128)

        self.r3_conv1 = nn.Conv2d(128, 128, 1, bias=False)
        self.r3_bn1 = nn.BatchNorm2d(128)
        self.r3_conv2 = nn.Conv2d(128, 128, 1, bias=False)
        self.r3_bn2 = nn.BatchNorm2d(128)

        self.r4_conv1 = nn.Conv2d(128, 128, 1, bias=False)
        self.r4_bn1 = nn.BatchNorm2d(128)
        self.r4_conv2 = nn.Conv2d(128, 128, 1, bias=False)
        self.r4_bn2 = nn.BatchNorm2d(128)
        """

        self.conv3 = nn.ConvTranspose2d(128, 64, 5, 2, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.ConvTranspose2d(64, 64, 4, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 3, 3, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        z = x
        y = self.r1_conv1(z)
        y = self.r1_bn1(y)
        y = self.relu(y)
        y = self.r1_conv2(y)
        y = self.r1_bn2(y)
        x = z + y

        """
        z = x
        y = self.r2_conv1(x)
        y = self.r2_bn1(y)
        y = self.relu(y)
        y = self.r2_conv2(y)
        y = self.r2_bn2(y)
        x = z + y

        z = x
        y = self.r3_conv1(x)
        y = self.r3_bn1(y)
        y = self.relu(y)
        y = self.r3_conv2(y)
        y = self.r3_bn2(y)
        x = z + y

        z = x
        y = self.r4_conv1(x)
        y = self.r4_bn1(y)
        y = self.relu(y)
        y = self.r4_conv2(y)
        y = self.r4_bn2(y)
        x = z + y
        """

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.sigmoid(x)

        return x

class Network:
    def __init__(self, device, dlr, dbeta, glr, gbeta):
        self.D = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 2, bias=False),
            nn.Sigmoid()
        ).to(device)
        self.G = Generator().to(device)
        self.G_optim = optim.Adam(self.G.parameters(), lr=glr, betas=(gbeta, 0.999))
        self.D_optim = optim.Adam(self.D.parameters(), lr=dlr, betas=(dbeta, 0.999))