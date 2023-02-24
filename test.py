import torch
import torchaudio
import librosa

from transforms_v2 import SpecAugment
from functional_v2 import warp_axis_torch, mask_along_axis
from utils.plots import plot_spectrogram



torch_data, sr  = torchaudio.load('audio_data/lex_30.wav')

# Convert audio array to mel spectrogram
librosa_mel = librosa.feature.melspectrogram(y=torch_data[0].numpy(),
                                             n_fft=2048,
                                             win_length=1024,
                                             sr=sr)

tensor_mel = torch.tensor(librosa_mel)
stacked_mel = tensor_mel.repeat((1000,1,1))

#augmented = warp_axis_torch(stacked_mel, 2, 1200)
#augmented = mask_along_axis(stacked_mel, 1, 3, 20)


spec = SpecAugment(
        warp_axis = 2,
        warp_w = 500,
        freq_mask_num = 3,
        freq_mask_param = 0,
        freq_mask_p = 0.1,
        time_mask_num = 3,
        time_mask_param = 0,
        time_mask_p = .05
)

augmented = spec(stacked_mel, 'torch')

#plot_spectrogram(augmented[1])
