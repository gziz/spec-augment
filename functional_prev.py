from typing import Optional
import math
import torch
from torch import Tensor
from torchaudio.functional import phase_vocoder


def time_stretch(complex_specgrams: Tensor, rate: float, hop_length: Optional[int] = None, n_freq: int = 201) -> Tensor:
    """
    Stretch stft in time without modifying pitch for a given rate.
    Args:
        complex_specgrams (Tensor): A tensor of dimension `(..., freq, num_frame)` with complex dtype.
        rate (float): speed up to apply
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        n_freq (int, optional): number of filter banks from stft. (Default: ``201``)

    Returns:
        Tensor:
            Stretched spectrogram. The resulting tensor is of the same dtype as the input
            spectrogram, but the number of frames is changed to ``ceil(num_frame / rate)``.
    """
    n_fft = (n_freq - 1) * 2
    hop_length = hop_length if hop_length is not None else n_fft // 2
    phase_advance = torch.linspace(0, math.pi * hop_length, n_freq).unsqueeze(-1)
    
    return phase_vocoder(complex_specgrams, rate, phase_advance)


def time_warp(specgram: Tensor, W: int = 0) -> Tensor:
    """
    Warp time axis between W boundaries, starting from point w0, the warp is either
    left or right, depending on the warp distance w, both randomly chosen.
    Args:
        specgram: A tensor `(..., freq, num_frame)`
        W: Boundary of time steps from within the warp takes place (W, n_frame - W)
    Returns:
        Tensor: Warped spectrogram
    """
    if specgram.dim() not in [2,3]:
        raise ValueError("Spectrogram should be at most three dimensional")

    if specgram.dim() == 2:
        specgram = specgram.unsqueeze(0)

    n_freq = specgram.shape[1]
    n_frames = specgram.shape[2]

    if W == 0:
        return specgram

    assert 2 * W < n_frames, (
        f"w {W} must be smaller than half the number of time frames {n_frames}")
    
    w0 = torch.randint(W, n_frames - W, ())
    w = torch.randint(-W + 1, W, ())

    left, right = specgram[:, :, :w0], specgram[:, :, w0:]

    left_rate = w0 / (w0 + w)
    right_rate = (n_frames - w0) / (n_frames - w0 - w)

    left = time_stretch(left, left_rate, n_freq=n_freq)
    right = time_stretch(right, right_rate, n_freq=n_freq)

    warped = torch.cat((left, right), axis=2)
    return warped



def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    if p == 1.0:
        return mask_param
    else:
        return min(mask_param, int(axis_length * p))


def mask_along_axis(
    specgram: Tensor,
    axis: int,
    n: int,
    mask_param: int,
    mask_value: float,
    p: float = 1.0,
) -> Tensor:
    """
    Apply mask along a spectrogram.
    The length of the mask is randomly chosen, with a cap on mask_param.
    Args
        specgram (tensor): Real spectrogram with dimensions (..., freq, time)
        axis (int): Masking is applied on (freq -> 1, time -> 2)
        n (int): Number of masks
        mask_param (int): Max length allowed for each individual mask.
        mask_value (float): Value for the masked portions
        p (float): Max proportion of masked rows/cols for each individual mask.
    Returns
        Tensor: Masked Tensor of dimensions (..., freq, time)
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram
    
    value = torch.rand(n) * mask_param

    mask_start = torch.rand(n) * (specgram.size(axis) - value)
    mask_start = mask_start.long()
    mask_end = (mask_start + value.long())

    mask_start = mask_start.view(n,1)
    mask_end = mask_end.view(n,1)

    masks = torch.arange(0, specgram.shape[axis]).repeat(n,1)
    masks = (masks >= mask_start) & (masks < mask_end)
    mask = masks.sum(dim=0).bool()
    
    if axis == 1:
        mask = mask.unsqueeze(-1)

    specgram = specgram.masked_fill(mask, mask_value)
    return specgram



def spec_augment(
    time_warp_w: int = 0,
    freq_mask_n: int = 0,
    freq_mask_f: int = 0,
    time_mask_n: int = 0,
    time_mask_t: int = 0,
    time_mask_p: float = 0.0,
    mask_value: float = 0.0
    ) -> Tensor:
    pass

