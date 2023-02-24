import numpy as np
import torch
from torch import Tensor


def warp_axis_torch(specgram:Tensor, axis:int, W:float):
    """
    Warp axis (frequency or time) between W boundaries, starting from point w0, the warp 
    direction can be negative or positive, depending on the randomly chosen distance w.
    Args:
        specgram: A tensor with dimensions (batch, freq, time)
        axis: Axis where the warp takes place
        W: Boundary of time steps where the warp takes place (W, num_warped_axis - W)
    Returns:
        tensor: Warped spectrogram
    """
    
    if axis not in [1, 2]:
            raise ValueError("Only Frequency and Time masking are supported")
            
    num_warped = specgram.shape[axis]
    num_non_warped = specgram.shape[1 if axis == 2 else 2]

    if W == 0:
        return specgram
    assert 2 * W < num_warped, (
        f"W param {W} must be smaller than half the size of the warped axis {num_warped}")
    
    w0 = torch.randint(W, num_warped - W, ())
    w = torch.randint(-W + 1, W, ())

    if axis == 1:
        lower, upper = specgram[:, :w0, :], specgram[:, w0:, :]
        lower_sz, upper_sz = (w0 + w, num_non_warped), (num_warped - w0 - w, num_non_warped)
    else:
        lower, upper = specgram[:, :, :w0], specgram[:, :, w0:]
        lower_sz, upper_sz = (num_non_warped, w0 + w), (num_non_warped, num_warped - w0 - w)
    
    # interpolate receives 4D: (batch, channel, freq, time)
    lower = lower.unsqueeze(1)
    upper = upper.unsqueeze(1)

    lower = torch.nn.functional.interpolate(lower, size=lower_sz, mode='bilinear')
    upper = torch.nn.functional.interpolate(upper, size=upper_sz, mode='bilinear')
    
    lower.squeeze_(1)
    upper.squeeze_(1)

    specgram = torch.cat((lower, upper), axis=axis)
    return specgram


def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    if mask_param == 0:
        return int(axis_length * p)
    else:
        return min(mask_param, int(axis_length * p))

def mask_along_axis(
    specgram: Tensor,
    axis: int,
    num_masks: int,
    mask_param: int,
    p: float = 1.0,
    mask_value: float = 0.0
):
    """
    Apply mask along a spectrogram.
    The length of the mask is randomly chosen, with a cap on mask_param.
    Args
        specgram: A tensor with dimensions (batch, freq, time)
        axis: Masking is applied (freq -> 1, time -> 2)
        num_masks: Number of masks
        mask_param: Max length allowed for each individual mask.
        p: Max proportion of masked rows/cols for each individual mask.
        mask_value: Value for the masked portions
    Returns
        specgram: Masked tensor of dimensions (batch, freq, time)
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram
    
    value = torch.randint(mask_param, ())

    for _ in range(num_masks):
        mask_start = torch.randint(specgram.shape[axis] - value, ())
        #mask_start = int(np.random.rand() * (specgram.shape[axis] - value))
        mask_end = mask_start + value

        if axis == 1:
            specgram[:, mask_start: mask_end, :] = mask_value
        else:
            specgram[:, :, mask_start: mask_end] = mask_value

    return specgram




# Outdated
def warp_axis_cv2(specgram, axis:int, W:float):
    """
    Warp axis (frequency or time) between W boundaries, starting from point w0, the warp 
    direction can be negative or positive, depending on the randomly chosen distance w.
    Args:
        specgram: A tensor|array with dimensions (freq, time)
        axis: Axis where the warp is applied (freq -> 1, time -> 2)
        W: Boundary of time steps where the warp takes place (W, num_warped_axis - W)
    Returns:
        tensor | array: Warped spectrogram
    """
    num_warped = specgram.shape[axis]
    num_non_warped = specgram.shape[abs(axis-1)]

    if W == 0:
        return specgram
    assert 2 * W < num_warped, (
        f"W param {W} must be smaller than half the size of the warped axis {num_warped}")
    
    w0 = np.random.randint(W, num_warped - W)
    w = np.random.randint(-W + 1, W)

    if axis == 0:
        lower, upper = specgram[:w0, :], specgram[w0:, :]
        lower_sz, upper_sz = (num_non_warped, w0 + w), (num_non_warped, num_warped - w0 - w)
    else:
        lower, upper = specgram[:, :w0], specgram[:, w0:]
        lower_sz, upper_sz = (w0 + w, num_non_warped), (num_warped - w0 - w, num_non_warped)

    import cv2
    lower = cv2.resize(lower, dsize=lower_sz, interpolation=cv2.INTER_LINEAR)
    upper = cv2.resize(upper, dsize=upper_sz, interpolation=cv2.INTER_LINEAR,)
    specgram = np.concatenate((lower, upper), axis=axis)

    return specgram