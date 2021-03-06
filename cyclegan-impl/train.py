import datetime
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Network


class SingleLabelLoader(datasets.CIFAR100):
    def __init__(self, *args, target_label, **kwargs):
        super(SingleLabelLoader, self).__init__(*args, **kwargs)

        new_data = []
        new_targets = []
        for (data, target) in zip(self.data, self.targets):
            if target == target_label:
                new_data.append(data)
                new_targets.append(0)
        self.data = new_data
        self.targets = new_targets


def main(n_epoch, lambda_, lambda2, dlr, dbeta, glr, gbeta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs(os.path.join('../results/cyclegan-impl', start_time), exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float)
    ])
    apple_set = SingleLabelLoader('../datasets/cifar100', train=True, download=True, target_label=0, transform=transform)
    orange_set = SingleLabelLoader('../datasets/cifar100', train=True, download=True, target_label=53, transform=transform)

    apple = Network(device, dlr, dbeta, glr, gbeta)
    orange = Network(device, dlr, dbeta, glr, gbeta)

    apple_loader = DataLoader(apple_set, batch_size=64, shuffle=True)
    orange_loader = DataLoader(orange_set, batch_size=64, shuffle=True)

    loss = torch.nn.BCELoss()
    l1loss = torch.nn.L1Loss()

    d_losses = []
    g_losses = []
    cycle_losses = []
    ident_losses = []

    for epoch in range(n_epoch):
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        cycle_loss_sum = 0.0
        ident_loss_sum = 0.0
        for (_apple_real_imgs, _), (_orange_real_imgs, _) in zip(apple_loader, orange_loader):
            n_imgs = len(_apple_real_imgs)
            zeros = torch.zeros((n_imgs, 1, 3, 3)).to(device)
            ones = torch.ones((n_imgs, 1, 3, 3)).to(device)

            apple_real_imgs = _apple_real_imgs.to(device)
            orange_real_imgs = _orange_real_imgs.to(device)

            # Discriminator
            apple.D_optim.zero_grad()
            orange.D_optim.zero_grad()
            apple_fake_imgs = apple.G(orange_real_imgs).detach()
            orange_fake_imgs = orange.G(apple_real_imgs).detach()

            apple_real_results = apple.D(apple_real_imgs)
            apple_fake_results = apple.D(apple_fake_imgs)
            apple_real_loss = loss(apple_real_results, ones)
            apple_fake_loss = loss(apple_fake_results, zeros)
            apple_loss_sum = apple_real_loss + apple_fake_loss

            orange_real_results = orange.D(orange_real_imgs)
            orange_fake_results = orange.D(orange_fake_imgs)
            orange_real_loss = loss(orange_real_results, ones)
            orange_fake_loss = loss(orange_fake_results, zeros)
            orange_loss_sum = orange_real_loss + orange_fake_loss

            # Discriminator Step
            apple_loss_sum.backward()
            orange_loss_sum.backward()
            apple.D_optim.step()
            orange.D_optim.step()

            # Generator
            apple.G_optim.zero_grad()
            orange.G_optim.zero_grad()
            apple_fake_imgs = apple.G(orange_real_imgs)
            orange_fake_imgs = orange.G(apple_real_imgs)

            apple_fake_results = apple.D(apple_fake_imgs)
            orange_fake_results = orange.D(orange_fake_imgs)

            apple_fake_loss = loss(apple_fake_results, ones)
            orange_fake_loss = loss(orange_fake_results, ones)

            # Generator (Cycle)
            apple_cycle_imgs = apple.G(orange_fake_imgs)
            orange_cycle_imgs = orange.G(apple_fake_imgs)
            apple_cycle_loss = l1loss(apple_real_imgs, apple_cycle_imgs)
            orange_cycle_loss = l1loss(orange_real_imgs, orange_cycle_imgs)
            cycle_loss = apple_cycle_loss + orange_cycle_loss

            # Generator (Ident)
            apple_ident_imgs = apple.G(apple_real_imgs)
            orange_ident_imgs = orange.G(orange_real_imgs)
            apple_ident_loss = l1loss(apple_real_imgs, apple_ident_imgs)
            orange_ident_loss = l1loss(orange_real_imgs, orange_ident_imgs)
            ident_loss = apple_ident_loss + orange_ident_loss

            loss_sum = apple_fake_loss + orange_fake_loss + lambda_ * cycle_loss + lambda2 * ident_loss

            # Generator Step
            loss_sum.backward()
            apple.G_optim.step()
            orange.G_optim.step()

            d_loss_sum += apple_loss_sum.item() + orange_loss_sum.item()
            g_loss_sum += apple_fake_loss.item() + orange_fake_loss.item()
            cycle_loss_sum += cycle_loss.item()
            ident_loss_sum += ident_loss.item()

        d_losses.append(d_loss_sum)
        g_losses.append(g_loss_sum)
        cycle_losses.append(cycle_loss_sum)
        ident_losses.append(ident_loss_sum)

        print('epoch: {:3d}, d_loss_sum: {:.3f}, g_loss_sum: {:.3f}, cycle_loss_sum: {:.3f}, ident_loss_sum: {:.3f}'.format(epoch, d_loss_sum, g_loss_sum, cycle_loss_sum, ident_loss_sum))

        apple_1 = apple_real_imgs[0].to('cpu').detach().numpy()
        apple_2 = orange_fake_imgs[0].to('cpu').detach().numpy()
        apple_3 = apple_cycle_imgs[0].to('cpu').detach().numpy()

        orange_1 = orange_real_imgs[0].to('cpu').detach().numpy()
        orange_2 = apple_fake_imgs[0].to('cpu').detach().numpy()
        orange_3 = orange_cycle_imgs[0].to('cpu').detach().numpy()

        img_path = os.path.join('../results/cyclegan-impl', start_time, 'gen_{}.png'.format(epoch))
        img = np.concatenate([apple_1, apple_2, apple_3, orange_1, orange_2, orange_3], axis=2).transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img * 255)

        graph_path = os.path.join('../results/cyclegan-impl', start_time, 'graph.png')
        plt.plot(list(range(1, len(d_losses) + 1)), d_losses, label='d_loss')
        plt.plot(list(range(1, len(g_losses) + 1)), g_losses, label='g_loss')
        plt.plot(list(range(1, len(cycle_losses) + 1)), cycle_losses, label='c_loss')
        plt.plot(list(range(1, len(ident_losses) + 1)), ident_losses, label='i_loss')
        plt.legend()
        plt.savefig(graph_path)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--lambda', type=float, dest='lambda_', default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--dlr', type=float, default=2e-4)
    parser.add_argument('--dbeta', type=float, default=0.5)
    parser.add_argument('--glr', type=float, default=2e-4)
    parser.add_argument('--gbeta', type=float, default=0.5)

    args = parser.parse_args()
    main(args.epoch, args.lambda_, args.lambda2, args.dlr, args.dbeta, args.glr, args.gbeta)