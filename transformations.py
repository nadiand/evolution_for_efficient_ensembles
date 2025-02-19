import cv2
import numpy as np

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def apply_to_mask(flag):
    def decorator(func):
        func.apply_to_mask = flag
        return func

    return decorator


@apply_to_mask(False)
def adjust_contrast(batch_imgs, alpha=1.0, is_mask=False):
    batch_imgs = torch.Tensor(batch_imgs)
    r, g, b = batch_imgs.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(batch_imgs.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    mean = torch.mean(l_img, dim=(-3, -2, -1), keepdim=True)
    result = alpha*batch_imgs + (1.0-alpha)*mean
    result[result > 3] = 3
    result[result < -3] = -3

    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(True)  # Change to False if you don't want this applied to masks
def resize_by_factor(batch_imgs, factor=1.0, is_mask=False):
    resized_imgs = []
    for img in batch_imgs:
        new_size = (int(img.shape[1] * factor), int(img.shape[2] * factor))
        if is_mask:
            resized_img = F.resize(torch.Tensor(img), new_size, interpolation=InterpolationMode.NEAREST)
        else:
            resized_img = F.resize(torch.Tensor(img), new_size)
        resized_imgs.append(resized_img)

    result = np.array(resized_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)  # Brightness adjustment typically doesn't apply to masks
def adjust_brightness(batch_imgs, delta=1, is_mask=False):
    adjusted_imgs = []
    for img in batch_imgs:
        # Ensure values wrap around
        adjusted_img = img*delta
        adjusted_img[adjusted_img > 3] = 3
        adjusted_img[adjusted_img < -3] = -3
        adjusted_imgs.append(adjusted_img)

    result = np.array(adjusted_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result
