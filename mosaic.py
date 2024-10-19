import numpy as np
import sys
import cv2

from point_reader import read_points
from warp_img import compute_homography, warp_image, compute_warped_image_bb
from multi_resolution_blending import multi_resolution_blend_images
from scipy.ndimage import distance_transform_edt

def blend_images(img1_warped, img2, disp):
    dx, dy = disp
    shift_x = max(-dx, 0)
    shift_y = max(-dy, 0)

    # make panorama
    panorama_width  = max(img1_warped.shape[1] + dx, img2.shape[1]) + shift_x
    panorama_height = max(img1_warped.shape[0] + dy, img2.shape[0]) + shift_y
    img1_panorama = np.zeros((panorama_height, panorama_width, 3))
    img2_panorama = np.zeros((panorama_height, panorama_width, 3))

    # Define placement coordinates for img1_warped
    x_start = dx + shift_x
    y_start = dy + shift_y
    y_end= y_start + img1_warped.shape[0]
    x_end = x_start + img1_warped.shape[1]

    img1_panorama[y_start:y_end, x_start:x_end] = img1_warped
    img2_panorama[shift_y:shift_y + img2.shape[0], shift_x:shift_x + img2.shape[1]] = img2

    img1_pano_mask = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    img2_pano_mask = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    mask_img1 = np.any(img1_warped != 0, axis=2).astype(np.uint8)  # Mask where img1_warped has valid pixels
    mask_img2 = np.any(img2 != 0, axis=2).astype(np.uint8)         # Mask where img2 has valid pixels
    img1_pano_mask[y_start:y_end, x_start:x_end] = mask_img1
    img2_pano_mask[shift_y:shift_y + img2.shape[0], shift_x:shift_x + img2.shape[1]] = mask_img2
    img1_dist_mask = distance_transform_edt(img1_pano_mask)  # Distance from non-zero (foreground) to zero
    img2_dist_mask = distance_transform_edt(img2_pano_mask)  # Same for img2

    overlap_mask = img1_dist_mask > img2_dist_mask
    overlap_mask = overlap_mask.astype(np.float32)

    cv2.imwrite("results/TEST_p1.png", img1_panorama)
    cv2.imwrite("results/TEST_p2.png", img2_panorama)

    return multi_resolution_blend_images(img1_panorama, img2_panorama, overlap_mask, 2, 10)

def main():
    if (len(sys.argv) < 6):
        print("Usage: python <script_name.py> <out_name> <img1_path> <img1_points_path> <img2_path> <img2_points_path> <warped_img1_path>")
        return 
    
    out_name = sys.argv[1]
    img1_path = sys.argv[2]
    img1_points_path = sys.argv[3]
    img2_path = sys.argv[4]
    img2_points_path = sys.argv[5]
    warped_img1_path = sys.argv[6] if (len(sys.argv) == 7) else None

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_dim = (img1.shape[1], img1.shape[0])
    img2_dim = (img2.shape[1], img2.shape[0])

    img1_pts = read_points(img1_points_path, img1_dim)
    img2_pts = read_points(img2_points_path, img2_dim)

    # compute homography from img1 to img2
    H = compute_homography(img1_pts, img2_pts)

    # warp img1 to have the view of img2
    img1_warped = None
    if (warped_img1_path is None):
        print("warping img1")
        img1_warped = warp_image(img1, H, "nearest")
    else:
        img1_warped = cv2.imread(warped_img1_path)
        print("loaded warped img1")
        
    warp_dim, warp_disp = compute_warped_image_bb(img1, H)

    print("blending...")

    panorama = blend_images(img1_warped, img2, warp_disp)
    cv2.imwrite("results/" + out_name + ".png", panorama)

if __name__ == "__main__":
    main()