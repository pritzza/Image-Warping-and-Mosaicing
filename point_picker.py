import matplotlib.pyplot as plt
import sys
import os

from point_reader import write_points

def pick_points(im_path):
    print('Please click on the image to select points. Press Enter when finished.')

    if not os.path.exists(im_path):
        print(f"Error: The file '{im_path}' does not exist.")
        sys.exit(1)

    im = plt.imread(im_path)
    height, width = im.shape[:2]

    fig, ax = plt.subplots()
    ax.imshow(im)

    points = []
    
    while True:
        point = plt.ginput(1, timeout=-1)
        if not point:
            break
        x, y = point[0]
        normalized_point = (x / width, y / height)
        points.append(normalized_point)
        ax.plot(x, y, 'ro', markersize=5)
        fig.canvas.draw()

    plt.close()

    write_points(points, im_path)

    return points

def main():
    if len(sys.argv) != 2:
        print("Usage: python point_picker.py <image_filename>")
        sys.exit(1)

    image_path = sys.argv[1]
    pick_points(image_path)

if __name__ == "__main__":
    main()