import argparse
import datetime
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.modules.activation import LeakyReLU
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import ConvertImageDtype

def main(dlr, dbeta, glr, gbeta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    descriminator = nn.Sequential(
        nn.Conv2d(1, 4, 4, 2, bias=False),
        nn.LeakyReLU(0.2),
        nn.Conv2d(4, 8, 4, 2, bias=False),
        nn.LeakyReLU(0.2),
        nn.Conv2d(8, 16, 2, bias=False),
        nn.LeakyReLU(0.2),
        nn.Conv2d(16, 1, 4, bias=False),
        nn.LeakyReLU(0.2),
        nn.Flatten(),
        nn.Sigmoid()
    ).to(device)
    generator = nn.Sequential(
        nn.Unflatten(1, (1, 6, 6)),
        nn.ConvTranspose2d(1, 4, 3),
        nn.ReLU(),
        nn.ConvTranspose2d(4, 8, 4),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 16, 3, 2),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 4),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 1, 3),
        nn.Sigmoid(),
    ).to(device)

    g_optim = torch.optim.Adam(generator.parameters(), lr=glr, betas=(gbeta, 0.999))
    d_optim = torch.optim.Adam(descriminator.parameters(), lr=dlr, betas=(dbeta, 0.999))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float)
    ])
    train_set = datasets.MNIST('../datasets/mnist', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # BinaryCrossEntropy
    loss = nn.BCELoss().to(device)

    d_losses = []
    g_losses = []

    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs(os.path.join('../results/gan-impl', start_time), exist_ok=True)

    for epoch in range(50):
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        for (imgs, _) in train_loader:

            num_imgs = len(imgs)
            real_img = imgs.to(device)

            ones = torch.ones(num_imgs, 1).to(device)
            zeros = torch.zeros(num_imgs, 1).to(device)

            d_optim.zero_grad()

            # 生成と推論
            noise = torch.randn(num_imgs, 36).to(device)
            g_gen = generator(noise).reshape(-1, 28, 28).detach()
            d_real_out = descriminator(real_img)
            d_fake_out = descriminator(g_gen[:, None])

            # descriminatorの学習
            d_loss_fake = loss(d_fake_out, zeros)
            d_loss_real = loss(d_real_out, ones)
            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()
            d_optim.step()

            # generator用に再生成
            g_optim.zero_grad()
            noise = torch.randn(num_imgs, 36).to(device)
            g_gen = generator(noise).reshape(-1, 28, 28)

            # generatorの学習
            g_loss = loss(descriminator(g_gen[:, None]), ones)
            g_loss.backward()
            g_optim.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()


        print('epoch: {:3d}, d_loss:{:.3f}, gen_loss:{:.3f}'.format(epoch + 1, d_loss_sum / len(train_loader), g_loss_sum / len(train_loader)))

        noise = torch.randn(10, 36).to(device)
        g_gen = generator(noise).reshape(-1, 28, 28).to('cpu').detach().numpy() * 256

        for idx in range(len(g_gen)):
            out_img = g_gen[idx].astype(np.int32)
            img_path = os.path.join('../results/gan-impl', start_time, 'gen_{}_{}.png'.format(epoch, idx))
            cv2.imwrite(img_path, out_img)

        graph_path = os.path.join('../results/gan-impl', start_time, 'graph.png')
        plt.plot(list(range(1, len(d_losses) + 1)), d_losses, label='d_loss')
        plt.plot(list(range(1, len(g_losses) + 1)), g_losses, label='g_loss')
        plt.legend()
        plt.savefig(graph_path)
        plt.close()

        d_path = os.path.join('../results/gan-impl', start_time, 'model_des_{}_{}.pth'.format(device, epoch))
        torch.save(generator.state_dict(), d_path)
        g_path = os.path.join('../results/gan-impl', start_time, 'model_gen_{}_{}.pth'.format(device, epoch))
        torch.save(descriminator.state_dict(), g_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlr', type=float, default=1e-5)
    parser.add_argument('--dbeta', type=float, default=0.1)
    parser.add_argument('--glr', type=float, default=2e-4)
    parser.add_argument('--gbeta', type=float, default=0.5)

    args = parser.parse_args()
    main(args.dlr, args.dbeta, args.glr, args.gbeta)