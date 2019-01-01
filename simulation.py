#!/usr/bin/env python
import math
import random
from typing import Tuple, Any
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers


Point = namedtuple('Point', ['x', 'y'])


class Wall:
    def __init__(self, p1: Point, p2: Point):
        if p1 < p2:
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1

        # calc coefficients of line equation (y = ax + b) going through points
        self.a = (p1.y - p2.y) / (p1.x - p2.x)
        self.b = (p1.x * p2.y - p2.x * p1.y) / (p1.x - p2.x)

    def calc_dist_and_norm_vec(self, point: Point) -> Tuple[float, Any]:
        """Get shortest distance from given point to the wall
        and normal vector of the wall.

                     o (x2, y2)
                    /
                   /
         (x0, y0) o
                 /
                /         o point(x3, y3)
               /
              o (x1, y1)

        :param point: point for which calculate the shortest dist to the wall
        :return: shortest distance to the wall and wall normal vector
        """
        a, b = self.a, self.b
        x3, y3 = point.x, point.y

        # calc crossing point between wall and
        # orthogonal line going through (x3, y3)
        x0 = (a * y3 + x3 - a * b) / (a**2 + 1)
        y0 = (a**2 * y3 + a * x3 + b) / (a**2 + 1)

        dist = np.sqrt((x3 - x0)**2 + (y3 - y0)**2)
        norm_vec = np.array([(x3 - x0), (y3 - y0)]) / dist

        # shortest distance to the wall is distance to start/end point
        # if crossing point lies beyond them
        if x0 < self.p1.x:
            dist = np.sqrt((x3 - self.p1.x)**2 + (y3 - self.p1.y)**2)
        elif x0 > self.p2.x:
            dist = np.sqrt((x3 - self.p2.x)**2 + (y3 - self.p2.y)**2)

        return dist, norm_vec


class Pedestrian(object):
    def __init__(self, start=(0, 0), end=(0, 0), v_init=0):
        self.position = start
        self.end = end
        self.v = v_init
    
    def move(self, vector=0):
        if not self.has_arrived():
            if not vector:
                x = self.end[0] - self.position[0]
                y = self.end[1] - self.position[1]
                d = math.sqrt(x**2+y**2)
                if d > self.v:
                    v = self.v
                else:
                    v = d
                self.position = (self.position[0] + v * x/d, self.position[1] + v * y/d)

    def has_arrived(self):
        return math.sqrt((self.position[0] - self.end[0])**2 + (self.position[1] - self.end[1])**2) < 0.001

    def random(factor=0.01):
        return Pedestrian(
            start=(random.random(), random.random()),
            end = (random.random(), random.random()),
            v_init = (random.random()/2 + 0.75) * factor # Speed v*factor, where 0.75<v<1.25 and factor is inverese of expected max number of iterations
        )


# Create random pedetrians
pedestrian_quantity = 10
pedestrians = [Pedestrian.random() for _ in range(pedestrian_quantity)]

for p in pedestrians:
    print("Pedestrian: s {},{} e {},{} v {}".format(p.position[0], p.position[1], p.end[0], p.end[1], p.v))

# Move pedestrians 
#step = 0
#while not all([p.has_arrived() for p in pedestrians]):
    #for p in pedestrians:
        #p.move()
    #step += 1

#print("Finished after {} steps".format(step))

def run_simulation():
    # move pedestrains
    history = []
    while not all([p.has_arrived() for p in pedestrians]):
        xs = []
        ys = []
        for p in pedestrians:
            p.move()
            xs.append(p.position[0])
            ys.append(p.position[1])
        history.append((xs, ys))

    # animated giph
    fig, ax = plt.subplots(figsize=(16,14))
    pedestrian_colors = np.random.rand(len(pedestrians))

    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=0, ymax=1)

    def update(frame):
        xs, ys = frame
        ax.clear()
        ax.scatter(xs, ys, c=pedestrian_colors)

    ani = FuncAnimation(fig, update, frames=history, interval=20, repeat=False)
    ani.save(f'ani2.mp4', writer=writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800))
    plt.show()

    # static plot
    for i in range(len(pedestrians)):
        xs = [epoch[0][i] for epoch in history]
        ys = [epoch[1][i] for epoch in history]
        plt.plot(xs, ys)
    plt.show()

run_simulation()
