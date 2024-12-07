import cv2
import sys

from point_reader import read_points

from warp_img import compute_homography, warp_image

def rectify_image(img, points, dim):
    rect_points = read_points("res/points/grid.points", dim)
    H = compute_homography(points, rect_points)
    warped_img = warp_image(img, H, crop=False)
    warped_img = warped_img[0:dim[1], 0:dim[0]]
    return warped_img

def main():
    if (len(sys.argv) != 6):
        print("Usage: python <script_name.py> <out_name> <img1_path> <img1_points_path> <width> <height>")
        return 
    
    out_name = sys.argv[1]
    img_path = sys.argv[2]
    img_points_path = sys.argv[3]
    width = int(sys.argv[4])
    height = int(sys.argv[5])

    img = cv2.imread(img_path)
    img_dim = (img.shape[1], img.shape[0])
    img_pts = read_points(img_points_path, img_dim)
    rectified_dim = (width, height)
    rectified_img = rectify_image(img, img_pts, rectified_dim)
    cv2.imwrite("results/" + out_name + ".png", rectified_img)

if __name__ == "__main__":
    main()