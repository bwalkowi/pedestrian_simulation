#!/usr/bin/env python
import random
import math

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
pedestrians = [ Pedestrian.random() for _ in range(pedestrian_quantity) ]

for p in pedestrians:
    print("Pedestrian: s {},{} e {},{} v {}".format(p.position[0], p.position[1], p.end[0], p.end[1], p.v))

# Move pedestrians 
step = 0
while not all([p.has_arrived() for p in pedestrians]):
    for p in pedestrians:
        p.move()
    step += 1

print("Finished after {} steps".format(step))
