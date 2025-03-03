import cv2
import numpy as np
from scipy.signal import convolve2d

import torch
import torchvision.transforms.functional as F
import torch.nn.functional as FF
from torchvision.transforms import InterpolationMode

LBOUNDS = torch.tensor([[[-2.1179039301310043]], [[-2.0357142857142856]], [[-1.8044444444444445]]])
UBOUNDS = torch.tensor([[[2.2489082969432315]], [[2.428571428571429]], [[2.6399999999999997]]])
MEAN=torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
STD=torch.tensor([[[0.229]], [[0.224]], [[0.225]]])

def apply_to_mask(flag):
    def decorator(func):
        func.apply_to_mask = flag
        return func

    return decorator

@apply_to_mask(False)
def adjust_contrast(batch_imgs, alpha=1.0, is_mask=False):
    adjusted_imgs = []
    for img in batch_imgs:
        r, g, b = img.unbind(dim=0)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
        mean = torch.mean(l_img, dim=(-3, -2, -1), keepdim=True)
        result = alpha*img + (1.0-alpha)*mean
        result = torch.clamp(result, LBOUNDS, UBOUNDS)
        adjusted_imgs.append(result)

    return torch.Tensor(np.array(adjusted_imgs))

@apply_to_mask(False)  # Brightness adjustment typically doesn't apply to masks
def adjust_brightness(batch_imgs, delta=0, is_mask=False):
    adjusted_imgs = []
    for img in batch_imgs:
        # Ensure values wrap around
        adjusted_img = img + delta*torch.ones_like(img)
        adjusted_img = torch.clamp(adjusted_img, LBOUNDS, UBOUNDS)
        adjusted_imgs.append(adjusted_img)

    return torch.Tensor(np.array(adjusted_imgs))

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

    return torch.Tensor(np.array(resized_imgs))

@apply_to_mask(False)
def adjust_gamma(batch_imgs, gamma=1.0, is_mask=False, min=0, max=1):
    inv_gamma = 1.0 / gamma

    adjusted_imgs = []
    for img in batch_imgs:
        adjusted_img = (STD*img + MEAN) ** inv_gamma
        adjusted_img = torch.clamp(adjusted_img, 0, 1)
        adjusted_imgs.append((adjusted_img - MEAN)/STD)

    return torch.Tensor(np.array(adjusted_imgs))

# def apply_sharpen_to_channels(img, kernel):
#     return np.array([convolve2d(channel, kernel, mode="same") for channel in img])

@apply_to_mask(False)
def apply_sharpen(batch_imgs, alpha=1.0, is_mask=False):
    # Base sharpening kernel
    a, b = -1.0, 9.0
    kernel = torch.tensor([[a, a, a], [a, b, a], [a, a, a]]).reshape(1, 3, 3).repeat(3, 1, 1)/9

    sharpened_imgs = []

    for img in batch_imgs:
        convolved_img = FF.conv2d(torch.unsqueeze((STD*img + MEAN), dim=0), torch.unsqueeze(kernel, dim=0), padding=1)
        sharpened_img = (STD*img + MEAN) * alpha + (1-alpha) * convolved_img
        sharpened_img = torch.squeeze((sharpened_img - MEAN)/STD, dim=0)
        sharpened_img = torch.clamp(sharpened_img, LBOUNDS, UBOUNDS)
        sharpened_imgs.append(sharpened_img)

    return torch.Tensor(np.array(sharpened_imgs))


def apply_blut(batch_imgs):
    new_imgs = F.gaussian_blur(batch_imgs, kernel_size=(5,9), sigma=(5,10))
    return new_imgs
