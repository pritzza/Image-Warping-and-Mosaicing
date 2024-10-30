import cv2
import sys
import numpy as np

from filters import compute_grad_mag, normalize_array, blur
from harris import get_harris_corners
from point_reader import write_points, normalize_points
from warp_img import compute_homography, warp_points, warp_image, compute_warped_image_bb
from mosaic import blend_images
from scipy.spatial import KDTree

def get_patch(img, point, dim):
    """
    Returns a patch of the image centered at 'point' with dimensions 'dim'.
    """
    assert len(point) == 2, "Point must be a tuple of (x, y)"
    assert len(dim) == 2, "Dim must be a tuple of (width, height)"
    
    x, y = point
    w, h = dim
    img_h, img_w = img.shape

    if not (0 <= x - (w // 2) and x + (w // 2) < img_w): # "Patch exceeds image width bounds"
        return None
    if not (0 <= y - (h // 2) and y + (h // 2) < img_h): #"Patch exceeds image height bounds"
        return None

    patch = img[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2]
    
    return patch

def get_processed_patches(points, grad_mag, patch_len=40):
    """
    Takes in points and gradient magnitude image, returns a list of
    normalized 8x8 patches.
    """
    assert len(points) > 0, "Must have at least one point"
    assert patch_len > 8,   "Patch length must be greater than 8"

    patches = []
    for p in points:
        patch = get_patch(grad_mag, p, (patch_len, patch_len))  # get 40x40 patch
        if patch is None:
            continue
        patch = normalize_array(patch)                          # bias/gain normalization
        patch = blur(patch, 4)                                  # downscale prep
        patch = cv2.resize(patch, (8, 8))                       # downscale
        patch = normalize_array(patch)                          # bias/gain normalization
        patches.append(patch)

    return patches

def get_2nn_matches(descriptors1, descriptors2, ratio_thresh=0.8):
    tree = KDTree(descriptors1)

    # distances: distances to the two nearest neighbors
    # indices: indices of the two nearest neighbors in descriptors1
    distances, indices = tree.query(descriptors2, k=2)

    matches = []
    
    for i in range(len(descriptors2)):
        n1_dist, n2_dist = distances[i]

        # if distances are close enough, match
        if n1_dist / n2_dist < ratio_thresh:
            matches.append((indices[i][0], i))  # (index in descriptors1, index in descriptors2)
            
    return matches

def RANSAC(points1, points2, eps=1.5, iterations=1000):
    assert len(points1) == len(points2), "Points must have the same length"
    assert len(points1) > 4, "At least 4 points are required"
    
    best_p1_inliers = []
    best_p2_inliers = []
    best_inlier_count = 0

    for _ in range(iterations):

        # get 4 random unique points
        rand_indices = np.random.choice(len(points1), 4, replace=False)
        rand_points1 = points1[rand_indices]
        rand_points2 = points2[rand_indices]

        # homography from img1 to img2
        H = compute_homography(rand_points1, rand_points2)

        # warp img1 points to plane of img2
        warped_points = warp_points(points1, H)

        p1_inliers = []
        p2_inliers = []
        inlier_count = 0
        for i in range(len(warped_points)):
            pw = warped_points[i]
            p = points2[i]
            
            dist = np.linalg.norm(pw - p)

            if dist < eps:
                inlier_count += 1
                p1_inliers.append(points1[i])
                p2_inliers.append(points2[i])

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_p1_inliers = p1_inliers
            best_p2_inliers = p2_inliers

    print("Best number of inliers:", best_inlier_count)
    best_H = compute_homography(best_p1_inliers, best_p2_inliers)
    return best_H

def main():
    if len(sys.argv) < 4:
        print("Usage: python auto_mosaic.py <out_name> <image1_path> <image2_path> ... <imageN_path>")
        sys.exit(1)

    out_name = sys.argv[1]
    image_paths = sys.argv[2:]
    imgs = [cv2.imread(path) for path in image_paths]
    
    for i in range(len(imgs)):
        assert imgs[i] is not None, f"Image {i} could not be read, check the path."
        assert imgs[i].shape == imgs[0].shape, "Images must be the same size."

    mosaics = [imgs[0]]

    PATCH_LEN = 40

    for i in range(1, len(imgs)):
        print(f"Processing image {i}")
        base_img = mosaics[-1]
        adding_img = imgs[i]

        base_img = np.clip(base_img, 0, 255).astype(np.uint8)
        adding_img = np.clip(adding_img, 0, 255).astype(np.uint8)

        #print("Base image shape: " + str(base_img.shape))
        #print("Adding image shape: " + str(adding_img.shape))

        # compute gradient magnitudes and Harris corners
        img1_grad_mag = compute_grad_mag(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY))
        img2_grad_mag = compute_grad_mag(cv2.cvtColor(adding_img, cv2.COLOR_BGR2GRAY))
        _, img1_points = get_harris_corners(img1_grad_mag, edge_discard=PATCH_LEN//2)
        _, img2_points = get_harris_corners(img2_grad_mag, edge_discard=PATCH_LEN//2)

        print(f"Found {len(img1_points)} points in image1 and {len(img2_points)} points in image2.")

        # process patches for matching
        patches1 = get_processed_patches(img1_points, img1_grad_mag)
        patches2 = get_processed_patches(img2_points, img2_grad_mag)
        flat_patches1 = np.array([patch.flatten() for patch in patches1])
        flat_patches2 = np.array([patch.flatten() for patch in patches2])

        # find matches using 2-NN matching
        matches = get_2nn_matches(flat_patches1, flat_patches2)

        print(f"Found {len(matches)} matches.")

        # get matched points from both images based on indices
        matched_img1_points = np.array([img1_points[i] for i, _ in matches])
        matched_img2_points = np.array([img2_points[j] for _, j in matches])

        # RANSAC feature matched points to get final homography for mosaicing
        H = RANSAC(matched_img2_points, matched_img1_points)

        print("warping...")

        # warp img1
        adding_img_warped = warp_image(adding_img, H)
        adding_img_warped = np.clip(adding_img_warped, 0, 255).astype(np.uint8)
        #cv2.imwrite(out_name + str(i) + "_warped.png", adding_img_warped)  # save intermdiate warped image
        #print("Warped image saved to " + out_name + "_warped.png")

        print("blending...")

        # perform mosaicing
        _, disp = compute_warped_image_bb(adding_img, H)
        blended_img = blend_images(adding_img_warped, base_img, disp)
        blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
        #cv2.imwrite(out_name + str(i) + ".png", blended_img)
        #print("Final blended image saved to " + out_name + ".png")

        mosaics.append(blended_img)

    # final result is going to be the last image in mosaics
    panorama = mosaics[-1]
    panorama[:, :, 3] = 255  # set alpha channel to 255
    cv2.imwrite(out_name + ".png", panorama)
    print("Final blended image saved to " + out_name + ".png")

main()