import math
from typing import Optional
import torch
from torch import Tensor

import prev_versions.functional_prev as F


class TimeStretch(torch.nn.Module):
    """Stretch stft in time without modifying pitch for a given rate.

    Args:
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        n_freq (int, optional): number of filter banks from stft. (Default: ``201``)
        fixed_rate (float or None, optional): rate to speed up or slow down by.
            If None is provided, rate must be passed to the forward method. (Default: ``None``)
    """

    def __init__(self, hop_length: Optional[int] = None, n_freq: int = 201, fixed_rate: Optional[float] = None) -> None:
        super(TimeStretch, self).__init__()

        self.fixed_rate = fixed_rate

        n_fft = (n_freq - 1) * 2
        hop_length = hop_length if hop_length is not None else n_fft // 2
        self.register_buffer("phase_advance", torch.linspace(0, math.pi * hop_length, n_freq)[..., None])


    def forward(self, complex_specgrams: Tensor, overriding_rate: Optional[float] = None) -> Tensor:
        """
        Args:
            complex_specgrams (Tensor):
                A tensor of dimension `(..., freq, num_frame)` with complex dtype.
            overriding_rate (float or None, optional): speed up to apply to this batch.
                If no rate is passed, use ``self.fixed_rate``. (Default: ``None``)

        Returns:
            Tensor:
                Stretched spectrogram. The resulting tensor is of the same dtype as the input
                spectrogram, but the number of frames is changed to ``ceil(num_frame / rate)``.
        """
        if overriding_rate is None:
            if self.fixed_rate is None:
                raise ValueError("If no fixed_rate is specified, must pass a valid rate to the forward method.")
            rate = self.fixed_rate
        else:
            rate = overriding_rate
        return torch.functional.phase_vocoder(complex_specgrams, rate, self.phase_advance)




class TimeWarp(torch.nn.Module):
    def __init__(self, W):
        super(TimeWarp, self).__init__()
        self.W = W
    def forward(self, specgram: Tensor):
        return F.time_warp(specgram, self.W)



class _AxisMasking(torch.nn.Module):

    def __init__(self, axis: int, n: int, mask_param: int,  mask_value: float = 0.0, p: float = 1.0) -> None:
        super(_AxisMasking, self).__init__()
        self.axis = axis
        self.n = n
        self.mask_param = mask_param
        self.mask_value = mask_value
        self.p = p

    def forward(self, specgram: Tensor) -> Tensor:
        return F.mask_along_axis(specgram, self.axis, self.n, self.mask_param, self.mask_value, p=self.p)


class FrequencyMasking(_AxisMasking):
    def __init__(self, mask_param: int, mask_n: int = 1, mask_value: float = 0.0) -> None:
        super(FrequencyMasking, self).__init__(1, mask_n, mask_param, mask_value)


class TimeMasking(_AxisMasking):
    def __init__(self,  mask_param: int, mask_n: int = 1, mask_value: float = 0.0, p: float = 1.0) -> None:
        super(TimeMasking, self).__init__(2, mask_n, mask_param, mask_value, p)




class SpecAugment(torch.nn.Module):
    """
    Data augmentation for spectrogram (audio)
    Args
        time_warp_w (int): Boundaries where time warp (W, N - W)
        freq_mask_n (int): Number of masks to apply to the frequency axis
        freq_mask_f (int): Max length of any individual frequency mask
        time_mask_n (int): Number of masks to apply to the time axis
        time_mask_t (int): Max length of any individual time mask
        time_mask_p (float): Max proportion that any individual time mask can have
    """
    def __init__(
        self,
        time_warp_w: int = 0,
        freq_mask_n: int = 0,
        freq_mask_f: int = 0,
        time_mask_n: int = 0,
        time_mask_t: int = 0,
        time_mask_p: float = 1.0,
        mask_value: float = 0.0
    ) -> None:
        super(SpecAugment, self).__init__()
        self.time_warp = TimeWarp(time_warp_w)
        self.freq_mask = FrequencyMasking(freq_mask_f, freq_mask_n, mask_value)
        self.time_mask = TimeMasking(time_mask_t, time_mask_n, mask_value, time_mask_p)
        
    def forward(self, specgram: Tensor):
        specgram = self.time_warp(specgram)

        if specgram.dim() == 2:
            specgram = specgram.unsqueeze(0)
        specgram = self.freq_mask(specgram)
        specgram = self.time_mask(specgram)
        return specgram



