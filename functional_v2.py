import numpy as np

def warp_axis(specgram, axis:int, W:float):
    """
    Warp axis (frequency or time) between W boundaries, starting from point w0, the warp 
    direction can be negative or positive, depending on the randomly chosen distance w.
    Args:
        specgram: A tensor|array with dimensions (freq, num_frame)
        axis: Axis where the warp takes place
        W: Boundary of time steps from where within the warp takes place (W, num_warped_axis - W)
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


def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    if mask_param == 0:
        return int(axis_length * p)
    else:
        return min(mask_param, int(axis_length * p))

def mask_along_axis_v2(
    specgram,
    axis: int,
    num_masks: int,
    mask_param: int = -1,
    p: float = 1.0,
    mask_value: float = 0.0
):
    """
    Apply mask along a spectrogram.
    The length of the mask is randomly chosen, with a cap on mask_param.
    Args
        specgram: A tensor|array with dimensions (freq, num_frame)
        axis: Masking is applied on (freq -> 0, time -> 1)
        num_masks: Number of masks
        mask_param: Max length allowed for each individual mask.
        p: Max proportion of masked rows/cols for each individual mask.
        mask_value: Value for the masked portions
    Returns
        specgram: Masked tensor|array of dimensions (freq, time)
    """
    if axis not in [0, 1]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram
    
    for _ in range(num_masks):
        
        value = int(np.random.rand() * mask_param)
        mask_start = int(np.random.rand() * (specgram.shape[axis] - value))
        mask_end = (mask_start + value)

        if axis == 0:
            specgram[mask_start: mask_end, :] = mask_value
        else:
            specgram[:, mask_start: mask_end] = mask_value

    return specgram
