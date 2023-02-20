import torch
import torchaudio
import librosa

from transforms_v2 import SpecAugment
from utils.local_fairseq import SpecAugmentTransform
from utils.plots import plot_spectrogram



torch_data, sr  = torchaudio.load('audio_data/lex_6.wav')
# Convert audio array to mel spectrogram
librosa_mel = librosa.feature.melspectrogram(y=torch_data[0].numpy(),
                                             n_fft=2048,
                                             win_length=1024,
                                             sr=sr)

### Current Implementation ###

sAug = SpecAugment(
            warp_axis=1,
            warp_w = 100,
            freq_mask_num = 2,
            freq_mask_param = 10,
            freq_mask_p = 1.0,
            time_mask_num = 3,
            time_mask_param = 50,
            time_mask_p = 1.0,
)
augmented = sAug(librosa_mel)

plot_spectrogram(augmented)
