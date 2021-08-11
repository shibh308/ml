import argparse
import datetime
import os
import numpy as np
import cv2
import torch
from torch import nn
from torch._C import Argument

def main(gen_path, num_imgs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = nn.Sequential(
        nn.Unflatten(1, (-1, 1, 1)),
        nn.ConvTranspose2d(100, 128, 3, 1, 0, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 2, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 2, 2, 2, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, 2, 2, 2, bias=False),
        nn.Sigmoid(),
    ).to(device)
    generator.load_state_dict(torch.load(gen_path))
    generator = generator.to(device)

    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    noise = torch.rand(num_imgs, 128).to(device)
    g_gen = generator(noise).reshape(-1, 28, 28)

    dir = '../results/gan-impl' if num_imgs == 1 else os.path.join('../results/gan-impl', 'img_' + start_time)
    os.makedirs(dir, exist_ok=True)

    for idx in range(num_imgs):
        img_path = os.path.join(dir, 'img_{}_{}.png'.format(start_time, idx))
        out_img = (g_gen[idx].to('cpu').detach().numpy() * 256).astype(np.int32)
        cv2.imwrite(img_path, out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gen_path', required=True)
    parser.add_argument('-n', '--num_imgs', type=int, default=1)

    args = parser.parse_args()
    main(args.gen_path, args.num_imgs)