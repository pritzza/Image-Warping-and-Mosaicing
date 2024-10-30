import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from point_reader import write_points, read_points

class PointEditor:
    def __init__(self, image_path, points):
        self.image_path = image_path
        self.points = points
        self.selected_point = None
        self.fig, self.ax = plt.subplots()
        self.image = plt.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        self.scatter = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.imshow(self.image)
        x = self.points[:, 0] * self.width
        y = self.points[:, 1] * self.height
        self.scatter = self.ax.scatter(x, y, c='r', s=10)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        cont, ind = self.scatter.contains(event)
        if cont:
            self.selected_point = ind['ind'][0]

    def on_motion(self, event):
        if self.selected_point is None:
            return
        if event.inaxes != self.ax:
            return
        self.points[self.selected_point] = (event.xdata / self.width, event.ydata / self.height)
        self.update_plot()

    def on_release(self, event):
        self.selected_point = None

    def on_key(self, event):
        if event.key == 'enter':
            plt.close(self.fig)

    def update_plot(self):
        x = self.points[:, 0] * self.width
        y = self.points[:, 1] * self.height
        self.scatter.set_offsets(np.c_[x, y])
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()

def edit_points(image_path, points_path):
    
    if os.path.exists(points_path):
        points = read_points(points_path)
    else:
        print(f"No existing points file found. Starting with an empty set of points.")
        points = np.array([])

    editor = PointEditor(image_path, points)
    editor.run()

    write_points(editor.points, points_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python .\point_editor.py <image_filename> <points_filename>")
        sys.exit(1)

    image_path = sys.argv[1]
    points_path = sys.argv[2]
    edit_points(image_path, points_path)

if __name__ == "__main__":
    main()