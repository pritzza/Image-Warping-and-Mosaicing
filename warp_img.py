import numpy as np
import sys
import cv2

from point_reader import read_points

# args are n*2 martices
def compute_homography(img1_pts, img2_pts):
    assert len(img1_pts) == len(img2_pts), "Number of points must match"
    assert len(img1_pts) >= 4, "At least 4 points are required"

    n = len(img1_pts)

    A = []
    b = []

    for i in range(n):
        x, y = img1_pts[i]
        x_prime, y_prime = img2_pts[i]

        # First equation (x' = ...)
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y])
        b.append(x_prime)

        # Second equation (y' = ...)
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y])
        b.append(y_prime)

    A = np.array(A)  # convert A to 2n x 8
    b = np.array(b)  # Convert b to 2n x 1

    # Solve for the unknown h vector using least squares (h has 8 elements)
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return H

def warp_points(points, H):
    points = np.array(points)
    warped_points = []
    for pt in points:
        p = np.array([pt[0], pt[1], 1])
        p_prime = np.dot(H, p)
        w = p_prime[2]
        warped_points.append((p_prime[0]/w, p_prime[1]/w))

    return np.array(warped_points)

def compute_warped_image_bb(img, H):
    height, width = img.shape[:2]
    tl = [0, 0]
    tr = [width, 0]
    bl = [0, height]
    br = [width, height]

    corners = [tl, tr, bl, br]
    warped_corners = warp_points(corners, H)

    x_coords = [corner[0] for corner in warped_corners]
    y_coords = [corner[1] for corner in warped_corners]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    w = max_x - min_x
    h = max_y - min_y

    dim = np.array([w, h], dtype=int)
    displacement = np.array([min_x, min_y], dtype=int)

    return dim, displacement

# TODO: update to use scipy.interpolate.griddata
import numpy as np

def warp_image(img, H, crop=True):
    # Add an alpha channel to the image if it doesn't already have one
    if img.shape[2] == 3:  # If input is RGB, add an alpha channel
        img = np.dstack([img, np.ones((img.shape[0], img.shape[1]), dtype=img.dtype) * 255])

    dim = (img.shape[1], img.shape[0])
    disp = (0, 0)
    if crop:
        dim, disp = compute_warped_image_bb(img, H)
    width, height = dim[0], dim[1]
    dx, dy = disp[0], disp[1]

    # Initialize an RGBA output image with full transparency
    warped_img = np.zeros((height, width, 4), dtype=img.dtype)  # 4 channels: RGBA

    H_inv = np.linalg.inv(H)

    for y in range(height):
        for x in range(width):
            p_prime = np.array([x + dx, y + dy, 1])
            p = np.dot(H_inv, p_prime)
            w = p[2]
            src_x, src_y = p[0] / w, p[1] / w

            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                # Nearest-neighbor sampling
                src_x, src_y = int(src_x), int(src_y)
                warped_img[y, x] = img[src_y, src_x]  # RGB channels

    return warped_img

def main():
    if (len(sys.argv) != 6):
        print("Usage: python warp_img.py <out_name> <img1_path> <img1_points_path> <img2_path> <img2_points_path>")
        return 
    
    out_name = sys.argv[1]
    img1_path = sys.argv[2]
    img1_points_path = sys.argv[3]
    img2_path = sys.argv[4]
    img2_points_path = sys.argv[5]

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_dim = (img1.shape[1], img1.shape[0])
    img2_dim = (img2.shape[1], img2.shape[0])

    img1_pts = read_points(img1_points_path, img1_dim)
    img2_pts = read_points(img2_points_path, img2_dim)

    # homography from img1 to img2
    H = compute_homography(img1_pts, img2_pts)
    
    print("warping...")

    # warp img1 to have view of img2
    img1_warped = warp_image(img1, H)

    cv2.imwrite(f"results/{out_name}.png", img1_warped)
    print(f"Saved warped image to results/{out_name}.png")

if __name__ == "__main__":
    main()