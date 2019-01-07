#!/usr/bin/env python
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt


# time within which pedestrian should reach his preferred speed
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
        """Gets shortest distance from given point to the wall
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
    def __init__(self, start, goal, *, v_init=None, r=2,
                 pref_speed=3, max_speed=5, safe_dist=2,
                 psychological_dist=2, anticipation_time=1,
                 pedestrians_to_avoid=3,
                 avoidance_min=1, avoidance_mid=5,
                 avoidance_max=8, avoidance_magnitude=1):
        """Creates pedestrian agent.

        :param start: pedestrian initial position
        :param goal: pedestrian destination
        :param v_init: pedestrian initial velocity
        :param r: radius (pedestrian is modeled as a disc)
        :param pref_speed: the speed at which pedestrian prefers to move
        :param max_speed: pedestrian maximal speed ( ||v|| <= max_speed)
        :param safe_dist: safe distance that pedestrian prefers to keep from walls
        :param psychological_dist: distance defining pedestrian personal space
        :param anticipation_time: time within which pedestrian resolves potential collisions
        :param pedestrians_to_avoid: number of pedestrians to avoid in first order
        """
        self.pos = start
        self.goal = goal
        self.r = r

        self.v = v_init if v_init is not None else np.array([0.0, 0.0])
        self.pref_speed = pref_speed
        self.max_speed = max_speed

        # needed to calculate avoidance force magnitude
        self.d_min = avoidance_min
        self.d_mid = avoidance_mid
        self.d_max = avoidance_max
        self.avoidance_magnitude = avoidance_magnitude

        # needed to calculate repulsive force
        self.safe_dist = safe_dist

        # needed to calculate evasive force
        self.psychological_dist = psychological_dist
        self.anticipation_time = anticipation_time
        self.pedestrians_to_avoid = pedestrians_to_avoid

    def move(self, force, dt):
        # TODO use integration
        self.v = self._limit_velocity(self.v + force * dt)
        self.pos += self.v * dt

    def calc_force(self, walls, pedestrians, dt):
        goal_force = self.calc_goal_force()
        repulsive_force = self.calc_repulsive_force(walls)
        evasive_force = self.calc_evasive_force(pedestrians, goal_force,
                                                repulsive_force, dt)
        return goal_force + repulsive_force + evasive_force

    def calc_goal_force(self):
        norm_vec = self.goal - self.pos
        norm_vec /= np.linalg.norm(norm_vec)

        return (1 / TAU) * (self.pref_speed * norm_vec - self.v)

    def calc_repulsive_force(self, walls):
        pos = Point(*self.pos)
        force = np.zeros(self.v.shape)
        for wall in walls:
            wall_dist, wall_norm_vec = wall.calc_dist_and_norm_vec(pos)
            if wall_dist - self.r < self.safe_dist:
                numerator = (self.safe_dist + self.r - wall_dist)
                denominator = (wall_dist - self.r)**STEEPNESS
                force += wall_norm_vec * numerator / denominator

        return force

    def calc_evasive_force(self, pedestrians, goal_force, repulsive_force, dt):
        v_des = self.v + (goal_force + repulsive_force) * dt
        v_des = self._limit_velocity(v_des)

        return self._calc_evasive_force_1(v_des, pedestrians)
        #return self._calc_evasive_force_2(v_des, pedestrians, dt)

    def _calc_evasive_force_1(self, v_des, pedestrians):
        force = np.zeros(self.v.shape)
        num = 0

        colliding_pedestrians = self._get_colliding_pedestrians(v_des,
                                                                pedestrians)
        avoidance_force = np.zeros(self.v.shape)
        for (collision_time, fst) in colliding_pedestrians:
            num += 1
            avoidance_force = self._calc_avoidance_force(v_des, fst, collision_time)
            force += avoidance_force / 2**num
        force += avoidance_force / 2**num

        return force

    def _calc_evasive_force_2(self, v_des, pedestrians, dt):
        force = np.zeros(self.v.shape)
        num = 0

        colliding_pedestrians = self._get_colliding_pedestrians(v_des,
                                                                pedestrians)
        while colliding_pedestrians:
            (collision_time, fst), *rest = colliding_pedestrians
            avoidance_force = self._calc_avoidance_force(v_des, fst,
                                                         collision_time)
            force += avoidance_force
            num += 1
            v_des = self._limit_velocity(v_des + avoidance_force * dt)

            rest = [other for col_time, other in rest]
            colliding_pedestrians = self._get_colliding_pedestrians(v_des, rest)

        if num > 0:
            return force / num
        else:
            return force

    def _get_colliding_pedestrians(self, v_des, pedestrians):
        colliding_pedestrians = []
        for pedestrian in pedestrians:
            collision_time = self._check_collision(v_des, pedestrian)
            if collision_time is not None:
                colliding_pedestrians.append((collision_time, pedestrian))
        colliding_pedestrians.sort(key=lambda t: t[0])
        return colliding_pedestrians[:self.pedestrians_to_avoid]

    def _calc_avoidance_force(self, v_des, other, collision_time):
        c_i = self.pos + collision_time * v_des
        c_j = other.pos + collision_time * other.v

        unit_vec = c_i - c_j
        unit_vec /= np.linalg.norm(unit_vec)

        dist = np.linalg.norm(c_i - self.pos) + np.linalg.norm(c_i - c_j) - self.r - other.r

        return unit_vec * self._avoidance_force_magnitude(dist)

    def _avoidance_force_magnitude(self, d):
        if d < self.d_min:
            return 1/(self.d_min-d) * self.avoidance_magnitude
        elif d < self.d_mid:
            return self.avoidance_magnitude
        elif d < self.d_max:
            return (1/(self.d_mid - self.d_max) * d - self.d_max / (self.d_mid - self.d_max)) * self.avoidance_magnitude
        else:
            return 0

    def _check_collision(self, v_des, pedestrian):
        if pedestrian is self:
            return None

        v = v_des - pedestrian.v
        x_ji = pedestrian.pos - self.pos

        a = np.sum(v**2)
        b = -2 * np.sum(x_ji * v)
        c = np.sum(x_ji**2) - (self.psychological_dist + pedestrian.r) ** 2

        delta = b**2 - 4 * a * c

        # if equation has no solution there is no collision
        if delta < 0:
            return None

        delta = np.sqrt(delta)

        t_1 = (-b - delta) / (2 * a)
        t_2 = (-b + delta) / (2 * a)

        if abs(t_1 - t_2) < 0.01:
            return None
        elif t_1 < 0 < t_2 or t_2 < 0 < t_1:
            return 0
        elif t_1 >= 0 and t_2 >= 0:
            collision_time = min(t_1, t_2)
            if collision_time <= self.anticipation_time:
                return collision_time
        else:
            return None

    def _limit_velocity(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return self.max_speed * velocity / speed
        else:
            return velocity

    def has_arrived(self):
        return np.linalg.norm(self.goal - self.pos) < 0.01


def run_simulation(pedestrians, walls, dt):
    # move pedestrains
    t = 0
    history = []
    while not all([p.has_arrived() for p in pedestrians]) and t < 500:
        xs = []
        ys = []
        forces = []
        for p in pedestrians:
            forces.append(p.calc_force(walls, pedestrians, dt))
        for p, force in zip(pedestrians, forces):
            p.move(force, dt)
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
        Pedestrian(start=np.array([10.0, 0.0]), goal=np.array([28.0, 20.0]))
    ]

    walls = [
        Wall(Point(0, 10), Point(20, 10))
    ]

    run_simulation(pedestrians, walls, 0.1)


def pedestrians_test():
    pedestrians = [
        Pedestrian(np.array([0.0, 0.0]), np.array([20.0, 20.0]), max_speed=10),
        Pedestrian(np.array([25.0, 0.0]), np.array([0.0, 20.0]))
    ]

    run_simulation(pedestrians, [], 0.1)


wall_test()
pedestrians_test()
