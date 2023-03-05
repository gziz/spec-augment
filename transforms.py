import torch
import functional as F


class SpecAugmentTransform(torch.nn.Module):
    """
    Data augmentation for spectrogram (audio)
    Args
        warp_axis: Axis where the warp takes place (0->freq, 1->time)
        warp_param: Boundaries where warp takes place (W, N - W), (W in paper)
        freq_mask_n: Number of masks to apply to the frequency axis, (mF in paper)
        freq_mask_param: Max length of any individual frequency mask, (F in paper)
        freq_mask_p: Max proportion that any individual freq mask can have
        time_mask_n: Number of masks to apply to the time axis, (mT in paper)
        time_mask_param: Max length of any individual time mask, (T in paper)
        time_mask_p: Max proportion that any individual time mask can have, (p in paper)
    """
    def __init__(
        self,
        warp_axis: int = 1,
        warp_param: int = 0,
        freq_mask_n: int = 0,
        freq_mask_param: int = 0,
        freq_mask_p: float = 1.0,
        time_mask_n: int = 0,
        time_mask_param: int = 0,
        time_mask_p: float = 1.0,
        mask_value: float = 0.0

    ) -> None:
        super(SpecAugmentTransform, self).__init__()
        self.warp_axis = warp_axis
        self.warp_param = warp_param
        self.freq_mask_n = freq_mask_n
        self.freq_mask_param = freq_mask_param
        self.freq_mask_p = freq_mask_p
        self.time_mask_n = time_mask_n
        self.time_mask_param = time_mask_param
        self.time_mask_p = time_mask_p
        self.mask_value = mask_value

    def forward(self, specgram: torch.Tensor):
        """
        Args 
            specgram: Tensor with dimensions (batch, frequency, time)
        Returns
            specgram: Augmented specgram tensor with dimensions (batch, frequency, time)
        """
        return  F.spec_augment(specgram, self.warp_axis, self.warp_param,
                               self.freq_mask_n, self.freq_mask_param, self.freq_mask_p,
                               self.time_mask_n, self.time_mask_param, self.time_mask_p,
                               self.mask_value)

    def libri_speech_basic(self):
        self.warp_param = 80
        self.freq_mask_param, self.freq_mask_n = 27, 1
        self.time_mask_param, self.time_mask_n = 100, 1
        self.time_mask_p = 1.0

    def libri_speech_double(self):
        self.warp_param = 80
        self.freq_mask_param, self.freq_mask_n = 27, 2
        self.time_mask_param, self.time_mask_n = 100, 2
        self.time_mask_p = 1.0

    def switchboard_mild(self):
        self.warp_param = 40
        self.freq_mask_param, self.freq_mask_n = 15, 2
        self.time_mask_param, self.time_mask_n = 70, 2
        self.time_mask_p = 0.2

    def switchboard_strong(self):
        self.warp_param = 40
        self.freq_mask_param, self.freq_mask_n = 27, 2
        self.time_mask_param, self.time_mask_n = 70, 2
        self.time_mask_p = 0.2