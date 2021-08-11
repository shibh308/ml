import argparse
import datetime
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import ConvertImageDtype

def main(d1, g1, g2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # [-1, ...] -> [-1, 1]
    descriminator = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, d1),
        nn.ReLU(),
        nn.Linear(d1, 1),
        nn.Sigmoid()
    ).to(device)
    # [-1, 128] -> [-1, 784]
    generator = nn.Sequential(
        nn.Linear(128, g1),
        nn.ReLU(),
        nn.Linear(g1, g2),
        nn.ReLU(),
        nn.Linear(g2, 784),
        nn.Sigmoid(),
    ).to(device)

    g_optim = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(descriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

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
            noise = torch.rand(num_imgs, 128).to(device)
            g_gen = generator(noise).reshape(-1, 28, 28).detach()
            d_real_out = descriminator(real_img)
            d_fake_out = descriminator(g_gen)

            # descriminatorの学習
            d_loss_fake = loss(d_fake_out, zeros)
            d_loss_real = loss(d_real_out, ones)
            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()
            d_optim.step()

            # generator用に再生成
            g_optim.zero_grad()
            noise = torch.rand(num_imgs, 128).to(device)
            g_gen = generator(noise).reshape(-1, 28, 28)

            # generatorの学習
            g_loss = loss(descriminator(g_gen), ones)
            g_loss.backward()
            g_optim.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()


        print('epoch: {:3d}, d_loss:{:.3f}, gen_loss:{:.3f}'.format(epoch + 1, d_loss_sum, g_loss_sum))

        noise = torch.rand(10, 128).to(device)
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
    parser.add_argument('--d1', type=int, default=512)
    parser.add_argument('--g1', type=int, default=512)
    parser.add_argument('--g2', type=int, default=1024)

    args = parser.parse_args()
    main(args.d1, args.g1, args.g2)