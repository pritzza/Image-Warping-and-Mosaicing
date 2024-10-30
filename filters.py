import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import sys

def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def normalize_to_8bit(arr):
    return (normalize_array(arr) * 255).astype(np.uint8)

def blur(img, sigma=2):
    ksize = int(6 * sigma) | 1  # ensure ksize is odd
    G = cv.getGaussianKernel(ksize=ksize, sigma=sigma)
    G2D = G @ G.T  # 2D Gaussian from outer product of 1D kernels
    
    return cv.filter2D(img, -1, G2D)

def compute_grad_mag(img):

    if (len(img.shape) == 3):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])
    horizontal_edges = convolve2d(img, Dx, mode='same')
    vertical_edges = convolve2d(img, Dy, mode='same')
    grad_mag = np.sqrt(horizontal_edges**2 + vertical_edges**2)

    #_, bin_grad = cv.threshold(normalize_to_8bit(grad_mag), 30, 255, cv.THRESH_BINARY)

    return grad_mag

def display_images(image_title_pairs):
    n = len(image_title_pairs)
    
    plt.figure(figsize=(4*n, 2*n))
    
    for i, (image, title) in enumerate(image_title_pairs, start=1):
        plt.subplot(1, n, i)
        plt.imshow(normalize_to_8bit(image), cmap='gray')
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    plt.show()

def process_image(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "Image could not be read, check the path."

    blurred = blur(img)
    grad_mag = compute_grad_mag(img)

    images = [
        (blurred, 'Blurred'),
        (grad_mag, 'Gradient Magnitude'),
    ]

    display_images(images)

def main():
    if len(sys.argv) != 2:
        print("Usage: python filters.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    
    process_image(img_path)

if __name__ == "__main__":
    main()
