import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def normalize_arr(arr):
    # Normalize the Laplacian image to range [0, 1]
    return cv.normalize(arr, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

def extend_to_3d(arr):
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    else:
        return arr

def to_01(arr):
    if (arr.max() > 1.0):
        arr = arr / 255.0
    return arr

def collapse(stack):
    return np.sum(stack, axis=0)

def low_pass(im, sigma):
    G = cv.getGaussianKernel(ksize=int(6 * sigma), sigma=sigma)
    G2D = G @ G.T 
    low_freq = cv.filter2D(im, -1, G2D)
    return low_freq

def gaussian_stack(im, n, sigma):
    gs_stack = [im]  # start with the original image
    for i in range(1, n):
        sigma_i = sigma * (2 ** i)  # increase sigma at each level
        blurred_im = low_pass(gs_stack[-1], sigma_i)
        gs_stack.append(blurred_im)
    return gs_stack

def laplacian_stack(gaussian_stack):
    ls_stack = []
    for i in range(len(gaussian_stack) - 1):
        laplacian = gaussian_stack[i] - gaussian_stack[i + 1]  # subtract consecutive levels
        ls_stack.append(laplacian)
    ls_stack.append(gaussian_stack[-1])  # append the last Gaussian level
    return ls_stack

def multi_resolution_blend_images(im1, im2, mask, layers, sigma):

    # assert that input standardized
    mask = extend_to_3d(mask)

    g1 = gaussian_stack(im1, layers, sigma)
    g2 = gaussian_stack(im2, layers, sigma)
    
    l1 = laplacian_stack(g1)
    l2 = laplacian_stack(g2)    

    masks = gaussian_stack(mask, layers, sigma)

    for i in range(len(masks)):
        l1[i] = l1[i] * masks[i]
        l2[i] = l2[i] * (1. - masks[i]) # inverted

    masked_im1 = collapse(l1)
    masked_im2 = collapse(l2)

    blended = masked_im1 + masked_im2

    b = cv.normalize(blended, None, 0, 255, cv.NORM_MINMAX)
    b = cv.cvtColor(b.astype(np.uint8), cv.COLOR_BGR2RGB)  
    cv.imwrite('res/blended.jpg', b)

    return blended
