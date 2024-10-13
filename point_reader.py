import numpy as np
import os

def is_normalized(points):
    points_array = np.array(points)
    return np.all((points_array >= 0) & (points_array <= 1))

def normalize_points(points, dim):

    points = np.array(points)

    if is_normalized(points):
        return points
    
    points[:, 0] *= 1/dim[0]
    points[:, 1] *= 1/dim[1]

    return points

def scale_points(points, dim):
    points = np.array(points)
    points[:, 0] *= dim[0]
    points[:, 1] *= dim[1]
    return points

def read_points(filename, dim=(1,1)):

    points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split(','))
                points.append((x, y))
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    points = np.array(points)
    points = scale_points(points, dim)

    if (dim == (1,1)):
        print("didnt normalize points")

    return points

def write_points(points, filename):
    im_name = os.path.splitext(os.path.basename(filename))[0]
    im_name = im_name.split('.')[0]
    points_filename = os.path.join("res/points", f"{im_name}.points")

    os.makedirs(os.path.dirname(points_filename), exist_ok=True)

    with open(points_filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]},{point[1]}\n")

    print(f"Points saved to {points_filename}")

