import numpy as np
import matplotlib.pyplot as plt

class Checker:

    def __init__(self, resolution = 2, tile_size = 1):
        # resolution: number of pixels in each dimension
        # tile_size: number of pixels individual tile has in each dimension
        # output: store the pattern

        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution should be dividable by 2 * tile_size.")

        self.checker_size = resolution // tile_size
        self.output = np.zeros((self.checker_size, self.checker_size))
        self.resolution = resolution
        self.tile_size = tile_size 


    # create the checkerboard pattern as a numpy array
    def draw(self):
        # Top left tile is black
        rows = np.arange(0, self.checker_size) % 2
        cols = np.arange(0, self.checker_size) % 2
        rows, cols = np.meshgrid(rows, cols)

        self.output = np.logical_xor(rows, cols).astype('int')

        self.output = np.repeat(self.output, self.tile_size, axis=0)
        self.output = np.repeat(self.output, self.tile_size, axis=1)

        return self.output.copy()

    def show(self):
        plt.axis('off')
        plt.imshow(self.draw(), cmap='gray')
        plt.show()



class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    # Create binary image of a circle as a numpy array
    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)

        x, y = np.meshgrid(x, y)

        self.output = ((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2) < (self.radius ** 2)

        return self.output.copy()

    def show(self):
        plt.axis('off')
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Spectrum():
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution, 3))

    def draw(self):
        self.output[:,:,0] = np.full((self.resolution, self.resolution), np.linspace(0, 1, self.resolution))
        self.output[:,:,1] = np.full((self.resolution, self.resolution), np.linspace(0, 1, self.resolution).reshape(self.resolution, 1))
        self.output[:,:,2] = np.full((self.resolution, self.resolution), np.linspace(1, 0, self.resolution))

        return self.output.copy()

    def show(self):
        plt.axis('off')
        plt.imshow(self.draw())
        plt.show()

checker = Checker(16, 4)
checker.show()


circle = Circle(1024, 200, (512, 256))
circle.show()

spectrum = Spectrum(256)
spectrum.show()