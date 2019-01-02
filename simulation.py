#!/usr/bin/env python
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt


# time within which agent should reach his preferred speed
TAU = 10

# steepness of the repulsive potential
STEEPNESS = 2


Point = namedtuple('Point', ['x', 'y'])


class Wall:
    def __init__(self, p1, p2):
        if p1 < p2:
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1

        # calc coefficients of line equation (y = ax + b) going through points
        self.a = (p1.y - p2.y) / (p1.x - p2.x)
        self.b = (p1.x * p2.y - p2.x * p1.y) / (p1.x - p2.x)

    def calc_dist_and_norm_vec(self, point):
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


class Pedestrian:
    def __init__(self, start, goal, v_init, r=2, pref_speed=3, max_speed=5,
                 safe_dist=2, psych_dist=2, anticipation_time=1,
                 pedestrians_to_avoid=3):
        """

        :param start: agent initial position
        :param goal: agent destination
        :param v_init: agent initial velocity
        :param r: agent radius (pedestrian is modeled as a disc)
        :param pref_speed: the speed at which agent prefers to move
        :param max_speed: agent maximal speed ( ||v|| < max_speed)
        :param safe_dist: safe distance that agent prefers to keep from buildings
        :param psych_dist: distance defining personal space of the pedestrian
        :param anticipation_time: time within which agent resolves potential collisions
        :param pedestrians_to_avoid: time within which agent resolves potential collisions
        """
        self.pos = start
        self.goal = goal
        self.r = r
        self.v = v_init
        self.v_des = v_init
        self.pref_speed = pref_speed
        self.max_speed = max_speed
        self.safe_dist = safe_dist
        self.psych_dist = psych_dist
        self.anticipation_time = anticipation_time
        self.pedestrians_to_avoid = pedestrians_to_avoid

    def calc_goal_force(self):
        norm_vec = self.goal - self.pos
        norm_vec /= np.linalg.norm(norm_vec)

        return (1 / TAU) * (self.pref_speed * norm_vec - self.v)

    def calc_repulsive_force(self, walls):
        force = np.zeros(self.v.shape)
        for wall in walls:
            wall_dist, wall_norm_vec = wall.calc_dist_and_norm_vec(Point(*self.pos))
            if wall_dist - self.r < self.safe_dist:
                numerator = (self.safe_dist + self.r - wall_dist)
                denominator = (wall_dist - self.r)**STEEPNESS
                force += wall_norm_vec * numerator / denominator

        return force

    def calc_v_des(self, walls, pedestrians, dt):
        if not self.has_arrived():
            goal_force = self.calc_goal_force()
            repulsive_force = self.calc_repulsive_force(walls)

            v_des = self.v + (goal_force + repulsive_force) * dt
            v_des = self._limit_velocity(v_des)

            self.v_des = self.avoid2(v_des, pedestrians)
        else:
            self.v_des = np.array([0, 0])

    def avoid1(self, v_des, pedestrians):
        on_collision = []
        for pedestrian in pedestrians:
            collision_time = self._check_collision(v_des, pedestrian)
            if collision_time is not None:
                on_collision.append((collision_time, pedestrian))

        on_collision.sort(key=lambda x: x[0])
        on_collision = on_collision[:self.pedestrians_to_avoid]
        #TODO

    def avoid2(self, v_des, pedestrians):
        on_collision = []
        for pedestrian in pedestrians:
            collision_time = self._check_collision(v_des, pedestrian)
            if collision_time is not None:
                on_collision.append((collision_time, pedestrian))

        while on_collision:
            (collision_time, fst), *rest = on_collision
            c_i = self.pos + collision_time * v_des
            c_j = fst.pos + collision_time * fst.v

            unit_vec = c_i - c_j
            unit_vec /= np.linalg.norm(unit_vec)

            D = np.linalg.norm(c_i - self.pos) + np.linalg.norm(c_i - c_j) - self.r - fst.r
            v_des = self._limit_velocity(v_des + (1/D)*unit_vec)

            on_collision = []
            for _, pedestrian in rest:
                collision_time = self._check_collision(v_des, pedestrian)
                if collision_time is not None:
                    on_collision.append((collision_time, pedestrian))

        return v_des

    def move(self, dt):
        self.v = self.v_des
        self.pos += self.v * dt

    def _limit_velocity(self, velocity):
        if np.linalg.norm(velocity) > self.max_speed:
            return self.max_speed * velocity / np.linalg.norm(velocity)
        else:
            return velocity

    def _check_collision(self, v_des, pedestrian):
        if pedestrian is self:
            return None

        v = v_des - pedestrian.v
        x_ji = pedestrian.pos - self.pos

        a = np.sum(v**2)
        b = -2 * np.sum(x_ji * v)
        c = np.sum(x_ji**2) - (self.psych_dist + pedestrian.r)**2

        delta = np.sqrt(b**2 - 4 * a * c)

        t_1 = (-b - delta) / (2 * a)
        t_2 = (-b + delta) / (2 * a)

        if t_1 < 0 < t_2 or t_2 < 0 < t_1:
            return 0
        elif t_1 >= 0 and t_2 >= 0:
            collision_time = min(t_1, t_2)
            if collision_time <= self.anticipation_time:
                return collision_time
        else:
            return None

    def has_arrived(self):
        return np.linalg.norm(self.goal - self.pos) < 0.001


def run_simulation(pedestrians, walls, dt):
    # move pedestrains
    t = 0
    history = []
    while not all([p.has_arrived() for p in pedestrians]) and t < 500:
        xs = []
        ys = []
        for p in pedestrians:
            p.calc_v_des(walls, pedestrians, dt)
        for p in pedestrians:
            p.move(dt)
            xs.append(p.pos[0])
            ys.append(p.pos[1])
        # for i, p in enumerate(pedestrians):
        #     print(i, p.pos)

        history.append((xs, ys))
        t += 1

    # static plot
    for wall in walls:
        xs = [wall.p1.x, wall.p2.x]
        ys = [wall.p1.y, wall.p2.y]
        plt.plot(xs, ys)

    for i in range(len(pedestrians)):
        xs = [epoch[0][i] for epoch in history]
        ys = [epoch[1][i] for epoch in history]
        plt.plot(xs, ys)
    plt.show()


def wall_test():
    pedestrians = [
        Pedestrian(start=np.array([10.0, 0.0]), goal=np.array([28.0, 20.0]),
                   v_init=np.array([0.0, 0.0]))
    ]

    walls = [
        Wall(Point(0, 10), Point(20, 10))
    ]

    run_simulation(pedestrians, walls, 0.1)


def pedestrians_test():
    pedestrians = [
        Pedestrian(np.array([0.0, 0.0]), np.array([20.0, 20.0]), np.array([0.0, 0.0]), max_speed=10),
        Pedestrian(np.array([25.0, 0.0]), np.array([0.0, 20.0]), np.array([0.0, 0.0]))
    ]

    run_simulation(pedestrians, [], 0.1)


# wall_test()
pedestrians_test()
