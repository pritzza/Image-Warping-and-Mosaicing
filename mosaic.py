
import numpy as np
import sys
import cv2

from point_reader import read_points
from warp_img import compute_homography, warp_image, compute_warped_image_bb
from multi_resolution_blending import multi_resolution_blend_images
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from filters import blur

def blend_images(img1_warped, img2, disp):
    dx, dy = disp
    shift_x = max(-dx, 0)
    shift_y = max(-dy, 0)

    # convert img2 to RGBA if it is RGB
    if img2.shape[2] == 3:
        # add an alpha channel set to 255 (fully opaque)
        img2 = np.dstack([img2, np.ones((img2.shape[0], img2.shape[1]), dtype=img2.dtype) * 255])

    # make panorama
    panorama_width  = max(img1_warped.shape[1] + dx, img2.shape[1]) + shift_x
    panorama_height = max(img1_warped.shape[0] + dy, img2.shape[0]) + shift_y
    img1_panorama = np.zeros((panorama_height, panorama_width, 4), dtype=img1_warped.dtype)
    img2_panorama = np.zeros((panorama_height, panorama_width, 4), dtype=img2.dtype)

    # Define placement coordinates for img1_warped
    x_start = dx + shift_x
    y_start = dy + shift_y
    y_end = y_start + img1_warped.shape[0]
    x_end = x_start + img1_warped.shape[1]
    
    img1_panorama[y_start:y_end, x_start:x_end] = img1_warped
    img2_panorama[shift_y:shift_y + img2.shape[0], shift_x:shift_x + img2.shape[1]] = img2

    img1_alpha_mask = (img1_panorama[:, :, 3] > 0).astype(np.uint8)
    img2_alpha_mask = (img2_panorama[:, :, 3] > 0).astype(np.uint8)
    img1_pano_mask = img1_alpha_mask
    img2_pano_mask = binary_fill_holes(img2_alpha_mask).astype(np.uint8)
    img1_dist_mask = distance_transform_edt(img1_pano_mask)
    img2_dist_mask = distance_transform_edt(img2_pano_mask)

    sigma = 10
    overlap_mask = (img1_dist_mask > img2_dist_mask).astype(np.float32)
    overlap_mask *= img1_alpha_mask
    #overlap_mask = blur(overlap_mask, sigma)

    return multi_resolution_blend_images(img1_panorama, img2_panorama, overlap_mask, 2, sigma)

def main():
    if (len(sys.argv) < 6):
        print("Usage: python mosaic.py <out_name> <img1_path> <img1_points_path> <img2_path> <img2_points_path> <warped_img1_path>")
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
        cv2.imwrite("results/" + out_name + "_warped.png", img1_warped)
    else:
        img1_warped = cv2.imread(warped_img1_path)
        print("loaded warped img1")
        
    warp_dim, warp_disp = compute_warped_image_bb(img1, H)

    print("blending...")

    panorama = blend_images(img1_warped, img2, warp_disp)
    cv2.imwrite("results/" + out_name + ".png", panorama)

if __name__ == "__main__":
    main()