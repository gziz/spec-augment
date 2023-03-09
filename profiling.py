import argparse

# from transforms import SpecAugmentTransform
# from functional import *
from utils.spectrogram import plot_specgram, create_specgram
from specaugment_torch import SpecAugmentTransform, spec_augment

batch_specgram = create_specgram("audio_data/lex_30.wav", batch_sz=100)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--paradigm", type=str, default="modular")
parser.add_argument("--loops", type=int, default=1)
args = parser.parse_args()

params = {
    "warp_axis": 2, 
    "max_warp_length": 200,
    "num_freq_mask": 0,
    "freq_max_mask_length":0,
    "freq_mask_max_proportion":0,
    "num_time_mask":0,
    "time_max_mask_length":0,
    "time_mask_max_proportion":0,
    "mask_value":0
}

if args.paradigm == "modular":
    spec = SpecAugmentTransform(**params)
    for _ in range(args.loops):
        augmented = spec(batch_specgram)

else:
    for _ in range(args.loops):
        augmented = spec_augment(batch_specgram, **params)

plot_specgram(augmented[0])
