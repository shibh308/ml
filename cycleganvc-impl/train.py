import argparse
import datetime
import glob
import os
import numpy as np
import pyworld as pw
import pysptk as sptk
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def calc_mcep(wav, fs, order):
    _f0, t = pw.dio(wav, fs)
    f0 = pw.stonemask(wav, _f0, t, fs)
    sp = pw.cheaptrick(wav, f0, t, fs)
    ap = pw.d4c(wav, f0, t, fs)
    mcep = sptk.sp2mc(sp, order=order, alpha=0.46)
    return mcep, f0, ap


class WavDataSet(Dataset):
    def __init__(self, data_pathes, transform=None, data_len=None, frame=128, mcep_only=True):
        super().__init__()
        self.transform = transform
        self.frame = frame
        self.mcep = []
        if not mcep_only:
            self.f0 = []
            self.ap = []
        l = len(data_pathes) if data_len is None else min(data_len, len(data_pathes))
        for path in data_pathes[:l]:
            data = np.load(path)
            assert data['fs'] == 22050
            assert(data['len'] == data['f0'].shape[0])
            self.mcep.append(data['mcep'])
            if not mcep_only:
                assert(data['len'] == data['mcep'].shape[0])
                assert(data['len'] == data['ap'].shape[0])
                self.f0.append(data['f0'])
                self.ap.append(data['ap'])

    def __len__(self):
        return len(self.mcep)

    def __getitem__(self, index):
        len = self.mcep[index].shape[0]
        st = np.random.randint(0, len - self.frame + 1)
        mcep = self.mcep[index][st:st+self.frame]
        print(mcep.shape)
        return self.transform(mcep)


def main(data_path, target_path, data_len, frame, n_epoch, lambda_, lambda2, dlr, dbeta, glr, gbeta):
    # start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # os.makedirs(os.path.join('../results/cycleganvc-impl', start_time), exist_ok=True)

    data_pathes = glob.glob(os.path.join(data_path, '*.npz'))
    target_pathes = glob.glob(os.path.join(target_path, '*.npz'))

    elm_set = WavDataSet(data_pathes, transform=transforms.ToTensor(), frame=frame, mcep_only=False, data_len=data_len)
    elm_loader = DataLoader(elm_set, batch_size=64, shuffle=True, num_workers=2)

    target_set = WavDataSet(target_pathes, transform=transforms.ToTensor(), frame=frame, mcep_only=False, data_len=data_len)
    target_loader = DataLoader(target_set, batch_size=64, shuffle=True, num_workers=2)

    # apple = Network(device, dlr, dbeta, glr, gbeta)

    # for i in range(100):
        # dataset[i]

    """
    sp = sptk.mc2sp(mcep, alpha = 0.46, fftlen = 1024)
    synthesized = pw.synthesize(f0, sp, ap, fs)
    soundfile.write('./inp.wav', wav, fs)
    soundfile.write('./out.wav', synthesized, fs)
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--data_len', type=int, default=None)
    parser.add_argument('--frame', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--lambda', type=float, dest='lambda_', default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--dlr', type=float, default=2e-4)
    parser.add_argument('--dbeta', type=float, default=0.5)
    parser.add_argument('--glr', type=float, default=2e-4)
    parser.add_argument('--gbeta', type=float, default=0.5)

    args = parser.parse_args()
    main(args.dataset, args.target, args.data_len, args.frame, args.epoch, args.lambda_, args.lambda2, args.dlr, args.dbeta, args.glr, args.gbeta)