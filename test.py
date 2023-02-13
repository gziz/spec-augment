import torch
import torchaudio
import librosa

from transforms import SpecAugment
from fairseq_spec import SpecAugmentTransform
from plots import plot_spectrogram



torch_data, sr  = torchaudio.load('lex_6.wav')
librosa_mel = librosa.feature.melspectrogram(y=torch_data[0].numpy(),
                                             n_fft=2048,
                                             win_length=1024,
                                             sr=sr)

### Current Implementation ###
librosa_mel = torch.tensor(librosa_mel)

sAug = SpecAugment(
            time_warp_w = 150,
            freq_mask_n = 2,
            freq_mask_f = 10,
            time_mask_n = 3,
            time_mask_t = 40,
            time_mask_p = 1.0,
)
augmented = sAug(librosa_mel)
if augmented.dim() == 3:
    augmented = augmented[0].squeeze()

### Fairseq ###
# fairseq_spec = \
#     SpecAugmentTransform(
#         time_warp_w = 150,
#         freq_mask_n = 2,
#         freq_mask_f = 40,
#         time_mask_n = 3,
#         time_mask_t = 10,
#         time_mask_p = 1.0,
# )
# augmented = fairseq_spec(librosa_mel)


plot_spectrogram(augmented)
