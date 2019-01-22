#!/usr/bin/env python
import random
import itertools
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
import matplotlib.patches as mpatches


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

        self.history = []

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
        self.v = self._limit_velocity(self.v + force * dt)
        self.pos += self.v * dt

    def calc_force(self, walls, pedestrians, dt):
        goal_force = self.calc_goal_force()
        repulsive_force = self.calc_repulsive_force(walls)
        evasive_force = self.calc_evasive_force(pedestrians, goal_force,
                                                repulsive_force, dt)
        force = goal_force + repulsive_force + evasive_force

        return force + self.calc_variation_force(force)

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
        # return self._calc_evasive_force_2(v_des, pedestrians, dt)

    def calc_variation_force(self, force):
        angle = np.radians(random.randint(1, 360))
        x0, y0 = force / 20

        x1 = np.cos(angle) * x0 + np.sin(angle) * y0
        y1 = np.sin(angle) * x0 + np.cos(angle) * y0

        return np.array([x1, y1])

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
        return np.linalg.norm(self.goal - self.pos) < 0.1


class Simulation:
    def __init__(self, pedestrians, walls,
                 dt=0.1, max_t=100, *,
                 y_lim=(-1, 30), x_lim=(-1, 30),
                 show_psychological_dist=True, rm_agent_on_arrival=True,
                 pedestrian_addition=0):

        self.pedestrians = pedestrians
        self.walls = walls
        self.dt = dt
        self.max_t = max_t
        self.t = 0

        self.show_psychological_dist = show_psychological_dist
        self.rm_agent_on_arrival = rm_agent_on_arrival
        self.pedestrian_addition = pedestrian_addition

        # Plot
        self.x_lim, self.y_lim = x_lim, y_lim
        self.fig = plt.figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, xlim=x_lim, ylim=y_lim)
        self.wall_data = self.ax.plot([], [], 'k-')

        if pedestrian_addition:
            self.active_pedestrians = pedestrians[:2]
            self.last_added = 0
            self.used_pedestrians = set(id(p) for p in self.active_pedestrians)
        else:
            self.active_pedestrians = pedestrians[:]

        self.pedestrian_patches = {}
        self.psychological_patches = {}

    def _step(self):
        if self.rm_agent_on_arrival:
            for p in self.active_pedestrians:
                if p.has_arrived():
                    self.active_pedestrians.remove(p)
                    self._remove_pedestrian(p)

        if self.pedestrian_addition and self.t - self.last_added > self.pedestrian_addition:
            for p in self.pedestrians:
                if id(p) not in self.used_pedestrians:
                    self.active_pedestrians.append(p)
                    self.used_pedestrians.add(id(p))
                    self._add_pedestrian(p)
                    self.last_added = self.t
                    break

        forces = []
        for p in self.active_pedestrians:
            if not p.has_arrived():
                forces.append(p.calc_force(self.walls, self.active_pedestrians, self.dt))
            else:
                forces.append(0)
        for p, force in zip(self.active_pedestrians, forces):
            p.move(force, self.dt)
            p.history.append(tuple(p.pos))

        self.t += self.dt

    def _add_pedestrian(self, p):
        c = mpatches.Circle(p.pos, radius=p.r, color='k', fill=True)
        self.ax.add_patch(c)
        self.pedestrian_patches[id(p)] = c

        if self.show_psychological_dist:
            c = mpatches.Circle(p.pos, radius=p.psychological_dist + p.r,
                                edgecolor='r', fill=False)
            self.ax.add_patch(c)
            self.psychological_patches[id(p)] = c

    def _remove_pedestrian(self, p):
        self.pedestrian_patches[id(p)].remove()
        if self.show_psychological_dist:
            self.psychological_patches[id(p)].remove()

    def _animate(self, _frame):
        self._step()

    def run(self):
        while self.t < self.max_t:
            self._step()

    def run_and_draw(self):
        self.run()
        self.draw()

    def live_run(self):
        for p in self.active_pedestrians:
            self._add_pedestrian(p)

        # Draw walls
        for wall in self.walls:
            path_data = [
                (Path.MOVETO, wall.p1),
                (Path.LINETO, wall.p2)
            ]
            cod, vert = zip(*path_data)
            path = Path(vert, cod)
            patch = mpatches.PathPatch(path, color='brown', linewidth=3)
            self.ax.add_patch(patch)

        _anim = FuncAnimation(self.fig, self._animate, interval=30,
                              frames=int(self.max_t / self.dt))
        plt.draw()
        plt.show()

    def draw(self, save_path=None):
        # Draw walls
        for wall in self.walls:
            xs = [wall.p1.x, wall.p2.x]
            ys = [wall.p1.y, wall.p2.y]
            plt.plot(xs, ys, color='brown', linewidth=3)

        # Draw traces of pedestrians
        for p in self.pedestrians:
            if p.history:
                xs, ys = zip(*p.history)
                plt.plot(xs, ys)

        plt.xlim(self.x_lim)
        plt.ylim(self.y_lim)

        if save_path:
            plt.savefig(save_path)
            plt.clf()
        else:
            plt.show()


def wall_test():
    pedestrians = [
        Pedestrian(start=np.array([10.0, 0.0]), goal=np.array([28.0, 20.0]))
    ]

    walls = [
        Wall(Point(0, 10), Point(20, 10))
    ]

    sim = Simulation(pedestrians, walls, 0.1)
    sim.live_run()
    sim.run_and_draw()


def pedestrians_test(**params):
    pedestrians = [
        Pedestrian(np.array([0.0, 0.0]), np.array([20.0, 20.0]), **params),
        Pedestrian(np.array([25.0, 0.0]), np.array([0.0, 20.0]), **params)
    ]

    sim = Simulation(pedestrians, [], 0.1)
    sim.live_run()
    sim.run_and_draw()


def symmetric_pedestrians_test():
    pedestrians = [
        Pedestrian(np.array([0.0, 0.0]), np.array([20.0, 20.0])),
        Pedestrian(np.array([20.0, 0.0]), np.array([0.0, 20.0]))
    ]

    sim = Simulation(pedestrians, [], 0.1)
    sim.live_run()
    sim.run_and_draw()


def hallway_test(size=25, passage_width=10, _save_path=None, **params):
    pedestrians = [
        Pedestrian(np.array([0.0, 5.0]), np.array([29.0, 5.0]), **params),
        Pedestrian(np.array([1.0, 2.0]), np.array([25.0, 9.0]), **params),
        Pedestrian(np.array([1.0, 6.0]), np.array([28.0, 1.0]), **params),
        Pedestrian(np.array([0.0, 8.0]), np.array([27.0, 3.0]), **params),
        Pedestrian(np.array([26.0, 2.0]), np.array([-1.0, 2.0]), **params),
        Pedestrian(np.array([25.0, 4.0]), np.array([-2.0, 8.0]), **params),
        Pedestrian(np.array([26.0, 6.0]), np.array([0.0, 4.0]), **params),
        Pedestrian(np.array([24.0, 8.0]), np.array([-2.0, 3.0]), **params),
    ]

    walls = [
        Wall(Point(0, 0), Point(size, 0)),
        Wall(Point(0, passage_width), Point(size, passage_width))
    ]

    sim = Simulation(pedestrians, walls, 0.1,
                     x_lim=(0, size), y_lim=(-1, passage_width + 1))
    sim.live_run()
    sim.run_and_draw()


def hallway_random_test(paramobj, pedestrian_num, size=40, passage_width=10,
                        spawn_width=5, pedestrian_addition=0):
    point_group_1 = RandomPoint(0., 1., spawn_width, passage_width - 1)
    point_group_2 = RandomPoint(size-spawn_width, 1., size, passage_width - 1)
    groups = [point_group_1, point_group_2]

    pedestrians = []
    for params in paramobj.random_param_list(pedestrian_num):
        random.shuffle(groups)
        p = Pedestrian(groups[0].get_point(), groups[1].get_point(), **params)
        pedestrians.append(p)

    walls = [
        Wall(Point(0, 0), Point(size, 0)),
        Wall(Point(0, passage_width), Point(size, passage_width))
    ]

    sim = Simulation(pedestrians, walls, 0.1,
                     x_lim=(-1, size + 1), y_lim=(-1, passage_width + 1),
                     pedestrian_addition=pedestrian_addition)
    sim.live_run()
    sim.run_and_draw()


def crossing_test(size=20, passage_width=5):

    ws = (size-passage_width)/2  # wall size
    e = 0.00001  # Fix for ZeroDivisionError
    walls = [
        # Bottom left corner
        Wall(Point(0, ws), Point(ws, ws)),
        Wall(Point(ws, 0), Point(ws-e, ws)),
        # Bottom right corner
        Wall(Point(size - ws, ws), Point(size, ws)),
        Wall(Point(size - ws, 0), Point(size-ws-e, ws)),
        # Top left corner
        Wall(Point(0, size-ws), Point(ws, size-ws)),
        Wall(Point(ws, size), Point(ws-e, size-ws)),
        # Top right corner
        Wall(Point(size - ws, size-ws), Point(size, size-ws)),
        Wall(Point(size-ws, size), Point(size-ws-e, size-ws)),
    ]

    pedestrians = [
        Pedestrian(np.array([0.0, 10.0]), np.array([10.0, 0.0])),
    ]

    sim = Simulation(pedestrians, walls, 0.1)
    sim.live_run()
    sim.run_and_draw()


def crossing_random_test(paramobj, pedestrian_num, size=40, passage_width=10,
                         spawn_width=5, pedestrian_addition=0):
    ws = (size-passage_width)/2  # wall size
    e = 0.00001  # Fix for ZeroDivisionError

    point_group_1 = RandomPoint(0., ws+1, spawn_width, ws + passage_width - 1)
    point_group_2 = RandomPoint(size-spawn_width, ws+1, size, ws + passage_width - 1)
    point_group_3 = RandomPoint(ws+1, 0., ws + passage_width - 1, spawn_width)
    point_group_4 = RandomPoint(ws+1, size - spawn_width, ws + passage_width - 1, size)

    groups = [[point_group_1, point_group_2], [point_group_3, point_group_4]]

    pedestrians = []
    for params in paramobj.random_param_list(pedestrian_num):
        random.shuffle(groups)
        random.shuffle(groups[0])
        p = Pedestrian(groups[0][0].get_point(), groups[0][1].get_point(), **params)
        pedestrians.append(p)

    walls = [
        # Bottom left corner
        Wall(Point(0, ws), Point(ws, ws)),
        Wall(Point(ws, 0), Point(ws-e, ws)),
        # Bottom right corner
        Wall(Point(size - ws, ws), Point(size, ws)),
        Wall(Point(size - ws, 0), Point(size-ws-e, ws)),
        # Top left corner
        Wall(Point(0, size-ws), Point(ws, size-ws)),
        Wall(Point(ws, size), Point(ws-e, size-ws)),
        # Top right corner
        Wall(Point(size - ws, size-ws), Point(size, size-ws)),
        Wall(Point(size-ws, size), Point(size-ws-e, size-ws)),
    ]

    sim = Simulation(pedestrians, walls, 0.1,
                     x_lim=(-1, size + 1), y_lim=(-1, size + 1),
                     pedestrian_addition=pedestrian_addition)
    sim.live_run()
    sim.run_and_draw()


class Param:
    def __init__(self, random_len=5, **params):
        """
        param values:
            list - predefined param list
            tuple
                (min_value, max_value) - will create a random list of size random_len
                (min_value, max_value, step) - values passed to np.arange
            other - will be passed as a single value list
        """
        self.params = params
        self.rlen = random_len
        self.generator = None

    def expand_params(self):
        expanded_params = {}
        for k, v in self.params.items():
            if isinstance(v, list):
                expanded_params[k] = v
            elif isinstance(v, tuple):
                if len(v) == 2:
                    expanded_params[k] = np.random.uniform(v[0], v[1], self.rlen).tolist()
                elif len(v) == 3:
                    expanded_params[k] = np.arange(*v)
                else:
                    raise RuntimeError("Dont know what to do with that many parameters in tuple")
            else:
                expanded_params[k] = [v]
        return expanded_params

    def param_space_generator(self):
        expanded_params = self.expand_params()
        for product in itertools.product(*self.expanded_params.values()):
            yield dict(zip(self.expanded_params.keys(), product))

    def random_param_list(self, n):
        expanded_params = {}
        for k, v in self.params.items():
            if isinstance(v, list):
                raise RuntimeError("List not supported for random params")
            elif isinstance(v, tuple):
                if len(v) == 2:
                    expanded_params[k] = np.random.uniform(v[0], v[1], n).tolist()
                else:
                    raise RuntimeError("Dont know what to do with that many parameters in tuple")
            else:
                expanded_params[k] = [v]*n
        return [dict(zip(expanded_params.keys(), l)) for l in zip(*expanded_params.values())]


class RandomPoint:
    def __init__(self, x1, y1, x2, y2):
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])

    def get_point(self):
        return np.random.random(2) * (self.p2 - self.p1) + self.p1


def param_space_test(test_fn, paramobj, save_path_prefix=""):
    for params in paramobj.param_space_generator():
        name = "{}/{}.png".format(save_path_prefix, '-'.join(["{}:{}".format(k, v)
                                                              for k, v in params.items()]))
        test_fn(save_path=name, **params)


# hallway_test(
#        r=0.3,
#        pref_speed=1.0,
#        max_speed=2.5,
#        safe_dist=0.8,
#        psychological_dist=1.2,
#        anticipation_time=4,
#        pedestrians_to_avoid=3,
#        avoidance_min=0.5,
#        avoidance_mid=5,
#        avoidance_max=8,
#        avoidance_magnitude=0.8,
# )
#


# hallway_random_test(
crossing_random_test(
        Param(
            r=(0.2, 0.3),
            pref_speed=(0.5, 1.5),
            max_speed=2.5,
            safe_dist=(0.5, 1.0),
            psychological_dist=(0.45, 1.20),
            anticipation_time=(4, 8),
            pedestrians_to_avoid=5,
            avoidance_min=0.5,
            avoidance_mid=5,
            avoidance_max=8,
            avoidance_magnitude=0.8,
        ),
        size=40,
        passage_width=10,
        pedestrian_num=40,
        pedestrian_addition=6
    )

# param_space_test(
#     hallway_test,
#     Param(
#         r=0.5,
#         pref_speed=(0.5, 1.5, 0.5),
#         max_speed=2.5,
#         safe_dist=(0.5, 1.1, 0.5),
#         psychological_dist=(0.3, 2, 0.3),
#         anticipation_time=[4, 8],
#         pedestrians_to_avoid=3,
#         avoidance_min=0.5,
#         avoidance_mid=5,
#         avoidance_max=8,
#         avoidance_magnitude=(0.5, 1.2, 0.3),
#         #size=30,
#         #passage_width=(5, 12, 3)
#     ),
#     save_path_prefix="param_space"
# )

# wall_test()

# pedestrians_test(
#         r=0.3,
#         pref_speed=1.0,
#         max_speed=2.5,
#         safe_dist=0.8,
#         psychological_dist=1.2,
#         anticipation_time=4,
#         pedestrians_to_avoid=3,
#         avoidance_min=0.5,
#         avoidance_mid=5,
#         avoidance_max=8,
#         avoidance_magnitude=0.8,
# )

# symmetric_pedestrians_test()

# hallway_test()

# crossing_test()
