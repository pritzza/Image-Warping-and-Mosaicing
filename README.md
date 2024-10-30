
# Image Processing and Mosaicing Project

This project provides tools for image processing, including Harrison Corner Detection, point selection, image warping, rectification, and mosaicing. It consists of several Python scripts that work together to automatically create panoramic images and perform related tasks.

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Overview Example](#overview-example)
   - [Point Picker](#point-picker)
   - [Warp Image](#warp-image)
   - [Image Mosaicing](#image-mosaicing)
   - [Auto Mosaic](#auto-mosaic)
   - [Image Rectification](#image-rectification)
4. [File Descriptions](#file-descriptions)

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/pritzza/Image-Warping-and-Mosaicing.git
   cd Image-Warping-and-Mosaicing
   ```

2. Install the required packages:
   ```bash
   pip install opencv-python numpy matplotlib scipy
   ```

## Usage

### Overview Example

For automatic mosaicing, simply run:
```bash
python auto_mosaic.py <out_name> <image1_path> <image2_path> ... <imageN_path>
```

For manual mosaicing, follow the steps below:

First, place one or two images in the 'res' folder. For example, we'll use `cal_l.png` and `cal_r.png`.

You'll need to set the point correspondences using `point_picker.py`.

We recommend setting the correspondences simultaneously because the order of the points matters.

Open a second terminal: in one terminal, run `python point_picker.py cal_l.png`, and in the other, run `python point_picker.py cal_r.png`.

Once you're done picking points, press Enter or close the window. The points will be saved to `res/points/cal_l.points` and `res/points/cal_r.points`.

Next, you want to warp the images using `warp_img.py`.

Run the following command:

```bash
python warp_img.py cal_warped res/cal_l.png res/points/cal_l.points res/cal_r.png res/points/cal_r.points
```

This will save the warped image as `results/cal_warped.png`.

If you want to skip the warping step, you can go straight to image mosaicing using `mosaic.py`. 

To do that, run:

```bash
python mosaic.py cal_mosaic res/cal_l.png res/points/cal_l.points res/cal_r.png res/points/cal_r.points
```

Or, if you want to reuse the warped image from the previous step to save time, run:

```bash
python mosaic.py cal_mosaic res/cal_l.png res/points/cal_l.points res/cal_r.png res/points/cal_r.points res/cal_warped.png
```

This will save the mosaiced image as `results/cal_mosaic.png`.

### Point Picker

The `point_picker.py` script allows you to select corresponding points on two images.

```bash
python point_picker.py <image_filename>
```

- Click on the image to select points.
- Press Enter when finished.
- Points will be saved as `res/points/<image_filename>.points`.

### Warp Image

The `warp_img.py` script warps one image to match the perspective of another.

```bash
python warp_img.py <out_name> <img1_path> <img1_points_path> <img2_path> <img2_points_path>
```

- The warped image will be saved as `results/<out_name>.png`.

### Image Mosaicing

The `mosaic.py` script creates a mosaic from two images with corresponding points explicitly specified.

```bash
python mosaic.py <out_name> <img1_path> <img1_points_path> <img2_path> <img2_points_path> [<warped_img1_path>]
```

- If `warped_img1_path` is provided, it will use the pre-warped image instead of computing it.
- The mosaiced image will be saved as `results/<out_name>.png`.

### Auto Mosaic

The `auto_mosaic.py` script automates the process of panoramic creation by allowing you to pass multiple images and automatically processes them into a single mosaic, no correspondence points required.

```bash
python auto_mosaic.py <out_name> <image1_path> <image2_path> ... <imageN_path>
```

- Replace `<out_name>` with the desired name for the output mosaic.
- Add as many image paths as needed to create a mosaic.
- The resulting mosaic will be saved as `results/<out_name>.png`.

### Image Rectification

The `rectify.py` script rectifies an image based on selected points.

```bash
python rectify.py <out_name> <img_path> <img_points_path> <width> <height>
```

- The rectified image will be saved as `results/<out_name>.png`.

## File Descriptions

- `point_picker.py`: GUI tool for selecting corresponding points on images.
- `warp_img.py`: Warps an image based on corresponding points.
- `mosaic.py`: Creates an image mosaic from two input images.
- `auto_mosaic.py`: Automates the creation of mosaics from multiple images.
- `rectify.py`: Rectifies an image based on selected points.
- `point_reader.py`: Utility for reading and writing point data.
- `multi_resolution_blending.py`: Implements multi-resolution blending for smooth image compositing.
