import random
import numpy as np
import math
import matplotlib.pyplot as plt
from multiagentgames.lib import util
import sys
random.seed(1234)
np.random.seed(1234)
''' 
Dataset collection in https://colab.research.google.com/drive/1E6vK8nmrixf-FiIxRmP0HjyJ1mj3kv6P
'''

@util.functiontable
class Dataset2D:
    def g8(size):
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        for i in range(size):
            point = np.random.randn(2) * .05
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        # dataset /= 1.414
        return dataset

    def g9(size):
        dataset = []
        for _ in range(size // 9):
            for x in range(-1, 2):
                for y in range(-1, 2):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset)
        return dataset

    def g16(size):
        scale = 2.
        centers = [(math.cos(k * 2 * math.pi / 16), math.sin(k * 2 * math.pi / 16)) for k in range(16)]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        for i in range(size):
            point = np.random.randn(2) * .05
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        # dataset /= 1.414
        return dataset

    def g24(size):
        scale = 2.
        centers = [(math.cos(k * 2 * math.pi / 16), math.sin(k * 2 * math.pi / 16)) for k in range(16)]
        centers.extend([(0.5 * math.cos(k * 2 * math.pi / 8), 0.5 * math.sin(k * 2 * math.pi / 8)) for k in range(8)])
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        for i in range(size):
            point = np.random.randn(2) * .05
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        # dataset /= 1.414
        return dataset

    def g25(size):
        dataset = []
        for _ in range(size // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset)
        # dataset /= 2.828  # stdev
        return dataset

    def g49(size):
        dataset = []
        for _ in range(size // 49):
            for x in range(-3, 4):
                for y in range(-3, 4):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset)
        # dataset /= 2.828  # stdev
        return dataset

def plot_data(dataset='g8'):
    data = Dataset2D[dataset](10000)
    x, y = data[:, 0], data[:, 1]
    plt.figure()
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    plot_data()