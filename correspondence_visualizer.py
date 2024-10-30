import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2

from point_reader import read_points  # Ensure this function reads points correctly from files

def plot_correspondences(img1, img2, points1, points2):
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Display the images side-by-side
    ax1.imshow(img1)
    ax1.axis('off')
    ax2.imshow(img2)
    ax2.axis('off')

    # Plot points with unique colors for each pair
    for (x1, y1), (x2, y2) in zip(points1, points2):
        # Generate a unique random color for each pair of points
        color = np.random.rand(3,)

        # Plot points on both images
        ax1.plot(x1, y1, 'o', color=color, markersize=5)  # Point in the first image
        ax2.plot(x2, y2, 'o', color=color, markersize=5)  # Corresponding point in the second image

    plt.show()

def main():
    if len(sys.argv) != 5:
        print("Usage: python .\correspondence_visualizer.py <image1_filename> <image2_filename> <points1_filename> <points2_filename>")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    points1_path = sys.argv[3]
    points2_path = sys.argv[4]

    # Load the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Images could not be read, check the paths.")
        sys.exit(1)

    # Convert images to RGB format for matplotlib display
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    dim = (img1.shape[1], img1.shape[0])

    # Load points from files using `read_points`
    points1 = []
    points2 = []

    if os.path.exists(points1_path) and os.path.exists(points2_path):
        points1 = read_points(points1_path, dim)
        points2 = read_points(points2_path, dim)
    else:
        print("Points files could not be found, check the paths.")
        sys.exit(1)

    # Display correspondences
    plot_correspondences(img1, img2, points1, points2)

if __name__ == "__main__":
    main()
