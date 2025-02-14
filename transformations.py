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
def apply_bilateral_filter(
    batch_imgs, d=1, sigma_color=1, sigma_space=1, is_mask=False
):
    filtered_imgs = [
        cv2.bilateralFilter(img, int(d), sigma_color, sigma_space) for img in batch_imgs
    ]
    result = np.array(filtered_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)
def apply_histogram_equalization(batch_imgs, apply=1.0, is_mask=False):
    if round(apply):
#        equalized_imgs = [cv2.equalizeHist(img) for img in batch_imgs]
        equalized_imgs = [F.equalize(img) for img in batch_imgs]
        result = np.array(equalized_imgs)
    else:
        result = np.array(batch_imgs)

    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)
def apply_blur(batch_imgs, kernel_size=5, sigma=0, is_mask=False):
    blurred_imgs = [
#        cv2.Gaussian(Blur(img, (kernel_size, kernel_size), sigma) for img in batch_imgs
        F.gaussian_blur(img, (kernel_size, kernel_size), sigma) for img in batch_imgs
    ]
    result = np.array(blurred_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)
def apply_sharpen(batch_imgs, alpha=1.5, is_mask=False):
    # Base sharpening kernel
#    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#    kernel = kernel * alpha
#    sharpened_imgs = [cv2.filter2D(img, -1, kernel) for img in batch_imgs]
    sharpened_imgs = [F.adjust_sharpness(torch.Tensor(img), alpha) for img in batch_imgs]
    result = np.array(sharpened_imgs)

    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)

    return result


@apply_to_mask(False)
def adjust_contrast(batch_imgs, alpha=1.0, is_mask=False):
 #   result = np.array([cv2.convertScaleAbs(img, alpha=alpha) for img in batch_imgs])
#    result = np.array([F.adjust_contrast(torch.Tensor(img), alpha) for img in batch_imgs])

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


@apply_to_mask(False)
def adjust_gamma(batch_imgs, gamma=1.0, is_mask=False):
#    inv_gamma = 1.0 / gamma
#    table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(np.uint8)
#    result = np.take(table, batch_imgs, axis=-1)
    result = [F.adjust_gamma(torch.Tensor(img), gamma) for img in batch_imgs]
    return np.array(result)


@apply_to_mask(True)
def rotate(batch_imgs, angle=90, is_mask=False):
    rotated_imgs = []
    for img in batch_imgs:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Choose interpolation method based on is_mask
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        rotated_img = cv2.warpAffine(img, M, (w, h), flags=interp)
        rotated_imgs.append(rotated_img)
    result = np.array(rotated_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(True)
def translate(batch_imgs, translate_x=0, translate_y=0, is_mask=False):
    translated_imgs = []
    for img in batch_imgs:
        (h, w) = img.shape[:2]

        # Translation matrix
        M = np.float32([[1, 0, translate_x], [0, 1, translate_y]])

        # Choose interpolation method based on is_mask
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        translated_img = cv2.warpAffine(img, M, (w, h), flags=interp)
#        translated_img = F.affine(img, ) ????
        translated_imgs.append(translated_img)
    result = np.array(translated_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(True)  # Change to False if you don't want this applied to masks
def resize_by_factor(batch_imgs, factor=1.0, is_mask=False):
    resized_imgs = []
    for img in batch_imgs:
#        img = img.numpy()
        new_size = (int(img.shape[1] * factor), int(img.shape[2] * factor))

        # Choose interpolation method based on is_mask
#        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
#        interp = torchvision.transforms.InterpolationMode.NEAREST if is_mask else

#        print(img)
#        print(img.shape, img.dtype)
#        resized_img = cv2.resize(img, new_size, interpolation=interp)
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
        # Ensure values wrap around in uint8 by using cv2.add
 #       adjusted_img = cv2.add(img, np.array([delta]))
#        adjusted_img = F.adjust_brightness(torch.Tensor(img), delta)
        adjusted_img = img*delta
        adjusted_img[adjusted_img > 3] = 3
        adjusted_img[adjusted_img < -3] = -3
        adjusted_imgs.append(adjusted_img)

    result = np.array(adjusted_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result

@apply_to_mask(False)
def solarize(batch_imgs, threshold, is_mask=False):
    adjusted_imgs = []
    for img in batch_imgs:
        # Ensure values wrap around in uint8 by using cv2.add
        adjusted_img = np.where(img>threshold, -img, img)
        adjusted_imgs.append(adjusted_img)

    result = np.array(adjusted_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result
