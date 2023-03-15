import argparse
import random

import torch

from utils.spectrogram import plot_specgram, create_specgram
from specaugment_torch import SpecAugmentTransform, spec_augment

batch_specgram = create_specgram("audio_data/lex_30.wav", batch_sz=100)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--paradigm", type=str, default="modular")
parser.add_argument("--loops", type=int, default=1)
args = parser.parse_args()

params = {
    "stretch_axis": 2, 
    "max_stretch_length": 400,
    "num_freq_masks": 3,
    "freq_max_mask_length":10,
    "num_time_masks":3,
    "time_max_mask_length": 100,
}

random.seed(1)

if args.paradigm == "modular":
    spec = SpecAugmentTransform(**params)
    spec.training = True
    for _ in range(args.loops):
        augmented = spec(batch_specgram)
else:
    for _ in range(args.loops):
        augmented = spec_augment(batch_specgram, **params)

#torch.set_printoptions(precision=6, sci_mode=True)
plot_specgram(augmented[0])

