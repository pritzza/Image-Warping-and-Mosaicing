import numpy as np
import sys
import cv2

from point_reader import read_points, write_points

# args are n*2 martices
def compute_homography(im1_pts, im2_pts):
    assert len(im1_pts) == len(im2_pts)
    assert len(im1_pts) >= 4

    return compute_homography_4pts(im1_pts, im2_pts)

def compute_homography_4pts(im1_pts, im2_pts):
    assert len(im1_pts) == 4 and len(im2_pts) == 4
    assert im1_pts.shape == (4, 2) and im2_pts.shape == (4, 2)

    # Create the A matrix (8x8) and b vector (8x1)
    A = []
    b = []

    for i in range(4):
        x, y = im1_pts[i]       # Coordinates in image 1
        x_prime, y_prime = im2_pts[i]  # Corresponding coordinates in image 2

        # First equation (x' = ...)
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y])
        b.append(x_prime)

        # Second equation (y' = ...)
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y])
        b.append(y_prime)

    A = np.array(A)  # Convert A into a numpy array (8x8)
    b = np.array(b)  # Convert b into a numpy array (8x1)

    # Solve for the unknown h vector (8 elements, corresponding to h1, h2, ..., h8)
    h = np.linalg.solve(A, b)

    # Reshape the h vector into a 3x3 homography matrix, where the last element is 1
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return H

def warp_image(img, H):
    warped_img = np.zeros(img.shape, dtype=img.dtype)
    H_inv = np.linalg.inv(H)
    height, width = img.shape[:2]

    for y in range(height):
        for x in range(width):
            p_prime = np.array([x, y, 1])
            p = np.dot(H_inv, p_prime)
            w = p[2]
            src_x, src_y = p[0] / w, p[1] / w

            # make sure point in bounds
            if 0 <= src_x < width and 0 <= src_y < height:
                src_x, src_y = int(src_x), int(src_y)   # nearest-neighbor sampling
                warped_img[y, x] = img[src_y, src_x]

    return warped_img

def main():
    img1_path = sys.argv[1]
    img1_points_path = sys.argv[2]
    img2_path = sys.argv[3]
    img2_points_path = sys.argv[4]

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_dim = (img1.shape[1], img1.shape[0])
    img2_dim = (img2.shape[1], img2.shape[0])

    img1_pts = read_points(img1_points_path, img1_dim)
    img2_pts = read_points(img2_points_path, img2_dim)

    H = compute_homography(img1_pts, img2_pts)
    
    img1_warped = warp_image(img1, H)

    cv2.imwrite("res/warped.png", img1_warped)

    return
    H_1 = np.linalg.inv(H)

    transformed_points = []

    for pt in img1_pts:
        p = np.array([pt[0], pt[1], 1])
        p_prime = np.dot(H, p)
        w = p_prime[2]
        transformed_points.append((p_prime[0]/w, p_prime[1]/w))

    write_points(transformed_points, "res/points/transformed.points")

if __name__ == "__main__":
    main()