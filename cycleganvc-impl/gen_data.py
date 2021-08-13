import argparse
import glob
import os
import numpy as np
import librosa
import pyworld as pw
import pysptk as sptk


def calc_mcep(wav, fs, order):
    _f0, t = pw.dio(wav, fs)
    f0 = pw.stonemask(wav, _f0, t, fs)
    sp = pw.cheaptrick(wav, f0, t, fs)
    ap = pw.d4c(wav, f0, t, fs)
    mcep = sptk.sp2mc(sp, order=order, alpha=0.46)
    return mcep, f0, ap


def main(input_dir, output_dir, num):
    os.makedirs(output_dir, exist_ok=True)
    wav_pathes = glob.glob(os.path.join(input_dir, '*.wav'))
    for i, path in enumerate(wav_pathes):
        if i >= num:
            break
        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(output_dir, name + '.npz')):
            continue
        wav, fs = librosa.load(path)
        wav = wav.astype(np.float64)
        mcep, f0, ap = calc_mcep(wav, fs, order=34)
        print(mcep.shape)
        np.savez(os.path.join(output_dir, name + '.npz'), mcep=mcep, f0=f0, ap=ap, fs=fs, len=mcep.shape[0])
        print('{:4d} / {:4d} : {}'.format(i + 1, len(wav_pathes), name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-n', '--num', required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.num)