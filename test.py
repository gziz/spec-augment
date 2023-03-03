import argparse

from transforms import SpecAugment
from functional import *
from utils.spectrogram import plot_specgram, create_specgram


batch_specgram = create_specgram("audio_data/lex_30.wav", batch_sz = 100)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--paradigm", type=str, default="modular")
parser.add_argument("--loops", type=int, default=1)
args = parser.parse_args()

params = {
    "warp_axis": 2,
    "warp_param": 1000,
    "freq_mask_num": 3,
    "freq_mask_param": 30,
    "freq_mask_p": 0.1,
    "time_mask_num": 3,
    "time_mask_param": 500,
    "time_mask_p": 0.5
}


if args.paradigm == "modular":
    spec = SpecAugment(**params)
    for _ in range(args.loops):
        augmented = spec(batch_specgram)

else:
    for _ in range(args.loops):
        augmented = spec_augment(batch_specgram, **params)

plot_specgram(augmented[0])
