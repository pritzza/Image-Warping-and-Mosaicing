import numpy as np
import sys
import cv2

from point_reader import read_points

def lerp(a, b, t):
    return a + (b - a) * t

def bilinear_interpolate(x, y, img):
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    if x0 < 0 or y0 < 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
        return 0 

    top_interp = lerp(img[y0, x0], img[y0, x1], x - x0)
    bot_interp = lerp(img[y1, x0], img[y1, x1], x - x0)
    
    return lerp(top_interp, bot_interp, y - y0)

# args are n*2 martices
def compute_homography(im1_pts, im2_pts):
    assert len(im1_pts) == len(im2_pts), "Number of points must match"
    assert len(im1_pts) >= 4, "At least 4 points are required"

    n = len(im1_pts)

    # Create the A matrix (2n x 8) and b vector (2n x 1)
    A = []
    b = []

    for i in range(n):
        x, y = im1_pts[i]       # Coordinates in image 1
        x_prime, y_prime = im2_pts[i]  # Corresponding coordinates in image 2

        # First equation (x' = ...)
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y])
        b.append(x_prime)

        # Second equation (y' = ...)
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y])
        b.append(y_prime)

    A = np.array(A)  # Convert A into a numpy array (2n x 8)
    b = np.array(b)  # Convert b into a numpy array (2n x 1)

    # Solve for the unknown h vector using least squares (h has 8 elements)
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Reshape the h vector into a 3x3 homography matrix, where the last element is 1
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
def warp_image(img, H, sampling="nearest", crop=True):

    dim = (img.shape[1], img.shape[0])
    disp = (0, 0)
    if crop:
        dim, disp = compute_warped_image_bb(img, H)
    width, height = dim[0], dim[1]
    dx, dy = disp[0], disp[1]

    warped_img = np.zeros((height, width, img.shape[2]), dtype=img.dtype)

    H_inv = np.linalg.inv(H)

    for y in range(height):
        for x in range(width):
            p_prime = np.array([x + dx, y + dy, 1])
            p = np.dot(H_inv, p_prime)
            w = p[2]
            src_x, src_y = p[0] / w, p[1] / w

            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                if sampling == "nearest":
                    src_x, src_y = int(src_x), int(src_y)
                    warped_img[y, x] = img[src_y, src_x]
                elif sampling == "bilinear":
                    warped_img[y, x] = bilinear_interpolate(src_x, src_y, img)
                else:
                    raise ValueError(f"Unknown sampling method: {sampling}")

    return warped_img

def main():
    if (len(sys.argv) != 6):
        print("Usage: python <script_name.py> <out_name> <img1_path> <img1_points_path> <img2_path> <img2_points_path>")
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
    img1_warped = warp_image(img1, H, "nearest")

    cv2.imwrite(f"results/{out_name}.png", img1_warped)
    print(f"Saved warped image to results/{out_name}.png")

if __name__ == "__main__":
    main()