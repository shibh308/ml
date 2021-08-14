import argparse
import datetime
import glob
import os
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import pyworld as pw
import pysptk as sptk
import torch
from model import Network


def calc_mcep(wav, fs, order):
    _f0, t = pw.dio(wav, fs)
    f0 = pw.stonemask(wav, _f0, t, fs)
    sp = pw.cheaptrick(wav, f0, t, fs)
    ap = pw.d4c(wav, f0, t, fs)
    mcep = sptk.sp2mc(sp, order=order, alpha=0.46)
    return mcep, f0, ap


class WavDataLoader:
    def __init__(self, data_pathes,data_len=None, frame=128, mcep_only=True, batch_size=64):
        self.frame = frame
        self.batch_size = batch_size
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

    def batch(self):
        idxes = np.random.randint(0, len(self.mcep), self.batch_size)
        dat = np.zeros((self.batch_size, self.frame, 35), dtype=np.float32)
        for i, index in enumerate(idxes):
            siz = self.mcep[index].shape[0]
            st = np.random.randint(0, siz - self.frame + 1)
            dat[i] = self.mcep[index][st:st+self.frame]
        return torch.Tensor(dat.transpose(0, 2, 1))


def output(file_path, loader, net, index, device):
    mcep = loader.mcep[index]
    f0 = loader.f0[index]
    ap = loader.ap[index]

    cat_sp = None

    for i in range(0, len(mcep), 128):
        if i + 128 > len(mcep):
            mcep_inp = mcep[len(mcep) - 128 : len(mcep)]
        else:
            mcep_inp = mcep[i : i + 128]

        mcep_inp = torch.Tensor(mcep_inp[None].transpose(0, 2, 1)).to(device)
        mcep_out = np.ascontiguousarray(net.G(mcep_inp).to('cpu').detach().numpy()[0].transpose(1, 0).astype(np.float64))

        if i + 128 > len(mcep):
            mcep_out = mcep_out[i - (len(mcep) - 128):]
        sp = sptk.mc2sp(mcep_out, alpha=0.46, fftlen=1024)
        if cat_sp is None:
            cat_sp = sp
        else:
            cat_sp = np.vstack([cat_sp, sp])

    syn = pw.synthesize(f0, cat_sp, ap, 22050)
    soundfile.write(file_path, syn, 22050)


def main(path_A, path_B, data_len, frame, lambda_, lambda2, dlr, dbeta, glr, gbeta, write_wav):
    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs(os.path.join('../results/cycleganvc-impl', start_time), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    A_pathes = glob.glob(os.path.join(path_A, '*.npz'))
    B_pathes = glob.glob(os.path.join(path_B, '*.npz'))

    A_loader = WavDataLoader(A_pathes, frame=frame, mcep_only=not write_wav, data_len=data_len, batch_size=64)
    B_loader = WavDataLoader(B_pathes, frame=frame, mcep_only=not write_wav, data_len=data_len, batch_size=64)

    A_net = Network(device, dlr, dbeta, glr, gbeta)
    B_net = Network(device, dlr, dbeta, glr, gbeta)

    loss = torch.nn.BCELoss()
    l1loss = torch.nn.L1Loss()

    d_losses = []
    g_losses = []
    cycle_losses = []
    ident_losses = []

    iter = 0
    while True:
        iter += 1
        A_real_voices = A_loader.batch().to(device)
        B_real_voices = B_loader.batch().to(device)
        n_voices = len(A_real_voices)
        zeros = torch.zeros((n_voices, 1)).to(device)
        ones = torch.ones((n_voices, 1)).to(device)

        A_real_voices = A_real_voices.to(device)
        B_real_voices = B_real_voices.to(device)

        # Discriminator
        A_net.D_optim.zero_grad()
        B_net.D_optim.zero_grad()
        A_fake_voices = A_net.G(B_real_voices).detach()
        B_fake_voices = B_net.G(A_real_voices).detach()

        A_real_results = A_net.D(A_real_voices[:, None])
        A_fake_results = A_net.D(A_fake_voices[:, None])
        A_real_loss = loss(A_real_results, ones)
        A_fake_loss = loss(A_fake_results, zeros)
        A_loss_sum = A_real_loss + A_fake_loss

        B_real_results = B_net.D(B_real_voices[:, None])
        B_fake_results = B_net.D(B_fake_voices[:, None])
        B_real_loss = loss(B_real_results, ones)
        B_fake_loss = loss(B_fake_results, zeros)
        B_loss_sum = B_real_loss + B_fake_loss

        # Discriminator Step
        A_loss_sum.backward()
        B_loss_sum.backward()
        A_net.D_optim.step()
        B_net.D_optim.step()

        # Generator
        A_net.G_optim.zero_grad()
        B_net.G_optim.zero_grad()
        A_fake_voices = A_net.G(B_real_voices)
        B_fake_voices = B_net.G(A_real_voices)

        A_fake_results = A_net.D(A_fake_voices[:, None])
        B_fake_results = B_net.D(B_fake_voices[:, None])

        A_fake_loss = loss(A_fake_results, ones)
        B_fake_loss = loss(B_fake_results, ones)

        # Generator (Cycle)
        A_cycle_voices = A_net.G(B_fake_voices)
        B_cycle_voices = B_net.G(A_fake_voices)
        A_cycle_loss = l1loss(A_real_voices, A_cycle_voices)
        B_cycle_loss = l1loss(B_real_voices, B_cycle_voices)
        cycle_loss = A_cycle_loss + B_cycle_loss

        # Generator (Ident)
        A_ident_voices = A_net.G(A_real_voices)
        B_ident_voices = B_net.G(B_real_voices)
        A_ident_loss = l1loss(A_real_voices, A_ident_voices)
        B_ident_loss = l1loss(B_real_voices, B_ident_voices)
        ident_loss = A_ident_loss + B_ident_loss

        loss_sum = A_fake_loss + B_fake_loss + lambda_ * cycle_loss + lambda2 * ident_loss

        # Generator Step
        loss_sum.backward()
        A_net.G_optim.step()
        B_net.G_optim.step()

        d_loss = A_loss_sum.item() + B_loss_sum.item()
        g_loss = A_fake_loss.item() + B_fake_loss.item()
        cycle_loss = cycle_loss.item()
        ident_loss = ident_loss.item()

        d_losses.append(d_loss)
        g_losses.append(g_loss)
        cycle_losses.append(cycle_loss)
        ident_losses.append(ident_loss)

        print('iter: {:3d}, d_loss: {:.3f}, g_loss: {:.3f}, cycle_loss: {:.3f}, ident_loss: {:.3f}'.format(iter, d_loss, g_loss, cycle_loss, ident_loss))

        if iter % 100 == 0 and write_wav:
            A_file_path = os.path.join('../results/cycleganvc-impl', start_time, 'A_{}.wav'.format(iter))
            B_file_path = os.path.join('../results/cycleganvc-impl', start_time, 'B_{}.wav'.format(iter))
            output(A_file_path, B_loader, A_net, 0, device)
            print('write:', A_file_path)
            output(B_file_path, A_loader, B_net, 0, device)
            print('write:', B_file_path)

            graph_path = os.path.join('../results/cycleganvc-impl', start_time, 'graph.png')
            plt.plot(list(range(1, len(d_losses) + 1)), d_losses, label='d_loss')
            plt.plot(list(range(1, len(g_losses) + 1)), g_losses, label='g_loss')
            plt.plot(list(range(1, len(cycle_losses) + 1)), cycle_losses, label='c_loss')
            plt.plot(list(range(1, len(ident_losses) + 1)), ident_losses, label='i_loss')
            plt.legend()
            plt.savefig(graph_path)
            plt.close()

    """
    sp = sptk.mc2sp(mcep, alpha = 0.46, fftlen = 1024)
    synthesized = pw.synthesize(f0, sp, ap, fs)
    soundfile.write('./inp.wav', wav, fs)
    soundfile.write('./out.wav', synthesized, fs)
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_A', required=True)
    parser.add_argument('--data_B', required=True)
    parser.add_argument('--data_len', type=int, default=None)
    parser.add_argument('--frame', type=int, default=128)
    parser.add_argument('--lambda', type=float, dest='lambda_', default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--dlr', type=float, default=2e-4)
    parser.add_argument('--dbeta', type=float, default=0.5)
    parser.add_argument('--glr', type=float, default=2e-4)
    parser.add_argument('--gbeta', type=float, default=0.5)
    parser.add_argument('--write', action='store_true')

    args = parser.parse_args()
    main(args.data_A, args.data_B, args.data_len, args.frame, args.lambda_, args.lambda2, args.dlr, args.dbeta, args.glr, args.gbeta, args.write)