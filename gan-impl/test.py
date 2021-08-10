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
        nn.Linear(128, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 784),
        nn.Sigmoid(),
    )
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