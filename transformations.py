import cv2
import numpy as np
from scipy.signal import convolve2d

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
    r, g, b = batch_imgs[:,0], batch_imgs[:,1], batch_imgs[:,2]
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b)
    mean = torch.mean(l_img, dim=(1,2), keepdim=True)
    mean = np.expand_dims(mean, axis=1)
    result = alpha*batch_imgs + (1.0-alpha)*mean
    result[result > 3] = 3
    result[result < -3] = -3

    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return torch.Tensor(result)

@apply_to_mask(False)  # Brightness adjustment typically doesn't apply to masks
def adjust_brightness(batch_imgs, delta=0, is_mask=False):
    adjusted_imgs = []
    for img in batch_imgs:
        # Ensure values wrap around
        adjusted_img = img + delta
        adjusted_img[adjusted_img > 3] = 3
        adjusted_img[adjusted_img < -3] = -3
        adjusted_imgs.append(adjusted_img)

    result = np.array(adjusted_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return torch.Tensor(result)

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
    return torch.Tensor(result)

@apply_to_mask(False)
def adjust_gamma(batch_imgs, gamma=1.0, is_mask=False):
    batch_imgs = batch_imgs.numpy()
    inv_gamma = 1.0 / gamma
    batch_imgs = (10*(batch_imgs + 3)).astype(np.uint8)
    table = ((np.arange(0, 6, 0.1) / 6) ** inv_gamma * 6 - 3)
    result = np.take(table, batch_imgs, axis=-1)
    return torch.Tensor(result)

def apply_sharpen_to_channels(img, kernel):
    return np.array([convolve2d(channel, kernel, mode="same") for channel in img])

@apply_to_mask(False)
def apply_sharpen(batch_imgs, alpha=1.0, is_mask=False):
    # Base sharpening kernel
    a, b = -1.0, 9.0
    kernel = np.array([[a, a, a], [a, b, a], [a, a, a]])
    sharpened_imgs = [img * alpha + (1-alpha) * apply_sharpen_to_channels(img, kernel) for img in batch_imgs]

    result = np.array(sharpened_imgs)

    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)

    return torch.Tensor(result)
