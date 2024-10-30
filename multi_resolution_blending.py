import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def normalize_arr(arr):
    return cv.normalize(arr, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

def extend_to_4d(arr):
    if arr.ndim == 2:
        return np.stack([arr] * 4, axis=-1)  # Extended to RGBA
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            return np.concatenate([np.repeat(arr, 3, axis=2), np.ones_like(arr)], axis=2)
        elif arr.shape[2] == 3:
            return np.concatenate([arr, np.ones(arr.shape[:2] + (1,))], axis=2)
        elif arr.shape[2] == 4:
            return arr
    return arr

def to_01(arr):
    if (arr.max() > 1.0):
        arr = arr / 255.0
    return arr

def collapse(stack):
    return np.sum(stack, axis=0)

def low_pass(im, sigma):
    channels = cv.split(im)
    G = cv.getGaussianKernel(ksize=int(6 * sigma), sigma=sigma)
    G2D = G @ G.T
    
    processed_channels = []
    for channel in channels:
        processed = cv.filter2D(channel, -1, G2D)
        processed_channels.append(processed)
    
    return cv.merge(processed_channels)

def gaussian_stack(im, n, sigma):
    gs_stack = [im]
    for i in range(1, n):
        sigma_i = sigma * (2 ** i)
        blurred_im = low_pass(gs_stack[-1], sigma_i)
        gs_stack.append(blurred_im)
    return gs_stack

def laplacian_stack(gaussian_stack):
    ls_stack = []
    for i in range(len(gaussian_stack) - 1):
        laplacian = gaussian_stack[i] - gaussian_stack[i + 1]
        ls_stack.append(laplacian)
    ls_stack.append(gaussian_stack[-1])
    return ls_stack

def multi_resolution_blend_images(im1, im2, mask, layers, sigma):

    # ensure inputs are in correct format
    im1 = extend_to_4d(to_01(im1.astype(np.float32)))
    im2 = extend_to_4d(to_01(im2.astype(np.float32)))
    mask = extend_to_4d(to_01(mask.astype(np.float32)))

    # generate Gaussian and Laplacian pyramids
    g1 = gaussian_stack(im1, layers, sigma)
    g2 = gaussian_stack(im2, layers, sigma)
    
    l1 = laplacian_stack(g1)
    l2 = laplacian_stack(g2)    

    masks = gaussian_stack(mask, layers, sigma)

    # blend each level
    for i in range(len(masks)):
        l1[i] = l1[i] * masks[i]
        l2[i] = l2[i] * (1. - masks[i])

    masked_im1 = collapse(l1)
    masked_im2 = collapse(l2)
    blended = masked_im1 + masked_im2

    # normalize and maintain alpha
    rgb_channels = blended[:, :, :3]
    alpha_channel = blended[:, :, 3:]
    normalized_rgb = cv.normalize(rgb_channels, None, 0, 255, cv.NORM_MINMAX)
    normalized_alpha = cv.normalize(alpha_channel, None, 0, 255, cv.NORM_MINMAX)
    
    # reshape alpha to maintain 3D shape with single channel
    if normalized_alpha.ndim == 2:
        normalized_alpha = normalized_alpha[..., np.newaxis]
    
    final_result = np.concatenate([normalized_rgb, normalized_alpha], axis=2)
    
    return final_result.astype(np.uint8)