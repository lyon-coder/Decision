import math
import gc
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
from pyclothoids import Clothoid, SolveG2
from shapely.geometry import Polygon
from dataclasses import dataclass
from enum import Enum, unique
import copy
import time


@dataclass
class State:
    x: float
    y: float
    v: float
    theta: float
    a: float
    omeca: float


@dataclass
class Param:
    width = 8
    length = 4
    wheel_base = 1.73
    front_suspension = 2.29
    rear_suspension = 0.4
    max_steering_angle = 6.6323
    max_longitudinal_acc = 2.0
    max_lateral_acc = 0.5
    # d_cr = 0.945


@dataclass
class Vehicle:
    id: int
    state: State
    param: Param


# @unique
# class Actions(Enum):
#     MaintainAcc = (2.5, 0)
#     MaintainDec = (-2.5, 0)
#     MaintainBrake = (-5, 0)
#     TurnleftAcc = (2.5, math.pi / 4)
#     TurnleftDec = (-2.5, math.pi / 4)
#     TurnleftBrake = (-5, math.pi / 4)
#     TurnrightAcc = (2.5, -math.pi / 4)
#     TurnrightDec = (-2.5, -math.pi / 4)
#     TurnrightBrake = (-5, -math.pi / 4)

Actions = {"Acc": (2.5, 0),
           "Dec": (-2.5, 0),
           "Maintain": (0, 0),
           "Turnleft": (0, math.pi / 4),
           "Turnright": (0, -math.pi / 4)}


def StateTransform(current_state: State, a: float, omeca: float, delta_t: float) -> State:
    next_state = current_state
    next_state.x = current_state.x + current_state.v * \
        math.cos(current_state.theta) * delta_t
    next_state.y = current_state.y + current_state.v * \
        math.sin(current_state.theta) * delta_t
    v = max(current_state.v + a * delta_t, 0)
    next_state.v = min(10.0, v)

    next_state.theta = current_state.theta + omeca * delta_t
    next_state.a = a
    next_state.omeca = omeca
    return next_state


def calculate_corner_points(center_x, center_y, width, height, angle):
    # 将角度转换为弧度
    angle_rad = angle

    # 计算矩形的半宽度和半高度
    half_width = width / 2
    half_height = height / 2

    # 计算矩形的四个角点相对于中心点的坐标
    top_left_x = center_x - half_width * \
        math.cos(angle_rad) + half_height * math.sin(angle_rad)
    top_left_y = center_y - half_width * \
        math.sin(angle_rad) - half_height * math.cos(angle_rad)

    top_right_x = center_x + half_width * \
        math.cos(angle_rad) + half_height * math.sin(angle_rad)
    top_right_y = center_y + half_width * \
        math.sin(angle_rad) - half_height * math.cos(angle_rad)

    bottom_left_x = center_x - half_width * \
        math.cos(angle_rad) - half_height * math.sin(angle_rad)
    bottom_left_y = center_y - half_width * \
        math.sin(angle_rad) + half_height * math.cos(angle_rad)

    bottom_right_x = center_x + half_width * \
        math.cos(angle_rad) - half_height * math.sin(angle_rad)
    bottom_right_y = center_y + half_width * \
        math.sin(angle_rad) + half_height * math.cos(angle_rad)

    # 返回四个角点的坐标
    return [(top_left_x, top_left_y),
            (top_right_x, top_right_y),
            (bottom_right_x, bottom_right_y),
            (bottom_left_x, bottom_left_y)]


def GetVertice(id: int, param: Param, state: State, lat_safe_dist: float, lon_safe_dis: float) -> list:
    vertices = list()
    vertices = calculate_corner_points(
        state.x, state.y, param.width + 2 * lat_safe_dist, param.length + 2 * lon_safe_dis, state.theta)
    return vertices


def PolygonIntersection(polygon1: list, polygon2: list) -> list:
    """
    计算两多边形交集
    """
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    intersection_poly = poly1.intersection(poly2)
    return list(intersection_poly.exterior.coords)


def PolygonArea(polygon: list) -> float:
    '''
    计算多边形面积
    '''
    if len(polygon) < 3:
        return 0
    poly = Polygon(polygon)
    return poly.area


def CalOverlap(vertice1: list, vertice2: list) -> float:
    intersection_poly = PolygonIntersection(vertice1, vertice2)
    overlap_area = PolygonArea(intersection_poly)
    return overlap_area


def Cal2DEuclideanDist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Node:
    def __init__(self, data: Vehicle):
        self.data = data
        self.children = [None] * 9
        self.father = None

    def insert(self, index, node):
        if index < 0 or index >= 9:
            raise IndexError("Index out of range.")

        self.children[index] = node

    def dfs_traversal(self):
        result = []  # 存储遍历结果
        result.append(self.data)  # 将当前节点的数据添加到结果列表

        for child in self.children:
            if child is not None:
                result.extend(child.dfs_traversal())  # 递归遍历子节点，并将结果扩展到结果列表

        return result


def bfs_build(vehicle_type, ttime, delta_t, pending_current_nodes, action_types, pending_nodes_records):
    build_tree_time_start = time.time()
    pending_next_nodes = []
    for node in pending_current_nodes:
        for i in range(len(action_types)):
            action = action_types[i]
            """
            LLLRR -> x
            LLLKK -> Y
            """
            last_action_lat = node.data.state.omeca
            last_action_lon = node.data.state.a
            action_value = Actions[action]
            # if((action_value[0] > 0 and last_action_lon < 0) or
            #    (action_value[0] < 0 and last_action_lon > 0) or
            #    (action_value[1] < 0 and last_action_lat > 0) or
            #    (action_value[1] > 0 and last_action_lat < 0)):
            #     continue
            vehicle_tmp = copy.deepcopy(node.data)
            state_tmp = StateTransform(vehicle_tmp.state,
                                       action_value[0], action_value[1], delta_t)
            if(not IsOverlap(vehicle_type, state_tmp)):
                vehicle_tmp.state = state_tmp
                child = Node(vehicle_tmp)
                child.father = node
                node.insert(i, child)
                pending_next_nodes.append(child)
    ttime -= delta_t
    if(pending_next_nodes and ttime > 0):
        bfs_build(vehicle_type, ttime, delta_t,
                  pending_next_nodes, action_types, pending_nodes_records)
    pending_nodes_records.append(pending_next_nodes)
    build_tree_time_end = time.time()
    print("Build Tree Time is ", build_tree_time_end - build_tree_time_start)


def IsOverlap(vehilce_type, state):
    is_over_lap = False
    if(vehilce_type == "ego"):
        if((state.y > -50 and state.y < -10 and state.x > 0 and state.x < 10) or
           (state.y > 10 and state.y < 50 and state.x > 0 and state.x < 10) or
           (state.y > -10 and state.y < 10 and state.x < 10 and state.x > 0)):
            if(math.sin(state.theta) <= 0):
                is_over_lap = True
            else:
                is_over_lap = False
        else:
            is_over_lap = True
    if(vehilce_type == "car"):
        if((state.y > 10 and state.y < 50 and state.x > -10 and state.x < 0) or
           (state.y > -10 and state.y < 10 and state.x > -10) or
           (state.x > 10 and state.y > -10 and state.y < 0)):
            if(math.sin(state.theta) >= 0):
                is_over_lap = True
            else:
                is_over_lap = False
        else:
            is_over_lap = True
    return is_over_lap


class GenReflineByClothoid():
    def __init__(self, car_start_x, car_start_y, car_start_heading, car_start_curvature, car_end_x, car_end_y, car_end_heading, car_end_curvature,
                 ego_start_x, ego_start_y, ego_start_heading, ego_start_curvature, ego_end_x, ego_end_y, ego_end_heading, ego_end_curvature) -> None:
        self.car_start_x = car_start_x
        self.car_start_y = car_start_y
        self.car_start_heading = car_start_heading
        self.car_start_curvature = car_start_curvature

        self.car_end_x = car_end_x
        self.car_end_y = car_end_y
        self.car_end_heading = car_end_heading
        self.car_end_curvature = car_end_curvature

        self.car_length = 4.0
        self.car_width = 2.0

        self.ego_start_x = ego_start_x
        self.ego_start_y = ego_start_y
        self.ego_start_heading = ego_start_heading
        self.ego_start_curvature = ego_start_curvature

        self.ego_end_x = ego_end_x
        self.ego_end_y = ego_end_y
        self.ego_end_heading = ego_end_heading
        self.ego_end_curvature = ego_end_curvature

        self.ego_width = 1.06
        self.ego_length = 2.87

        self.min_lat_dis = 0.2

        self.left_x_bound = self.ego_end_x - 0.5 * self.ego_width - self.min_lat_dis
        self.right_x_bound = self.ego_end_x + 0.5 * self.ego_width + self.min_lat_dis

        self.resolution = 0.2

    def gen_ego_refline(self) -> dict:
        ego_refline = dict()

        total_length = 0.0
        ego_x_list = list()
        ego_y_list = list()
        ego_theta_list = list()
        ego_curvature_list = list()
        ego_s_list = list()
        clothoid_list = list()
        clothoid_length_list = list()

        clothoid2_ego = SolveG2(self.ego_start_x, self.ego_start_y, self.ego_start_heading, self.ego_start_curvature,
                                self.ego_end_x, self.ego_end_y, self.ego_end_heading, self.ego_end_curvature)

        for point in clothoid2_ego:
            clothoid_list.append(point)
            clothoid_length_list.append(point.length)
            total_length += point.length

        traj_num = int(np.ceil(total_length/self.resolution + 0.5))
        for i in range(traj_num):
            s = i * self.resolution
            if s > total_length:
                break
            else:
                if s > clothoid_length_list[0] + clothoid_length_list[1]:
                    x = clothoid_list[2].X(
                        s - clothoid_length_list[0] - clothoid_length_list[1])
                    y = clothoid_list[2].Y(
                        s - clothoid_length_list[0] - clothoid_length_list[1])
                    theta = clothoid_list[2].Theta(
                        s - clothoid_length_list[0] - clothoid_length_list[1])
                    curvature = self.ego_start_curvature + \
                        clothoid_list[0].dk * clothoid_length_list[0] + \
                        clothoid_list[1].dk * clothoid_length_list[1]
                    curvature = curvature + \
                        clothoid_list[2].dk * \
                        (s - clothoid_length_list[0] - clothoid_length_list[1])
                    ego_x_list.append(x)
                    ego_y_list.append(y)
                    ego_theta_list.append(theta)
                    ego_curvature_list.append(curvature)
                    ego_s_list.append(s)
                elif s > clothoid_length_list[0]:
                    x = clothoid_list[1].X(s - clothoid_length_list[0])
                    y = clothoid_list[1].Y(s - clothoid_length_list[0])
                    theta = clothoid_list[1].Theta(s - clothoid_length_list[0])
                    curvature = self.ego_start_curvature + \
                        clothoid_list[0].dk * clothoid_length_list[0] + \
                        clothoid_list[1].dk * (s - clothoid_length_list[0])
                    ego_x_list.append(x)
                    ego_y_list.append(y)
                    ego_theta_list.append(theta)
                    ego_curvature_list.append(curvature)
                    ego_s_list.append(s)
                else:
                    x = clothoid_list[0].X(s)
                    y = clothoid_list[0].Y(s)
                    theta = clothoid_list[0].Theta(s)
                    curvature = self.ego_start_curvature + \
                        clothoid_list[0].dk * s
                    ego_x_list.append(x)
                    ego_y_list.append(y)
                    ego_theta_list.append(theta)
                    ego_curvature_list.append(curvature)
                    ego_s_list.append(s)

        ego_refline["x_list"] = ego_x_list
        ego_refline["y_list"] = ego_y_list
        ego_refline["theta_list"] = ego_theta_list
        ego_refline["curvature_list"] = ego_curvature_list
        ego_refline["s_list"] = ego_s_list
        ego_refline["total_length"] = total_length
        return ego_refline

    def gen_car_refline(self) -> dict:
        car_refline = dict()

        total_length = 0.0
        car_x_list = list()
        car_y_list = list()
        car_theta_list = list()
        car_curvature_list = list()
        car_s_list = list()
        clothoid_list = list()
        clothoid_length_list = list()

        clothoid2_car = SolveG2(self.car_start_x, self.car_start_y, self.car_start_heading, self.car_start_curvature,
                                self.car_end_x, self.car_end_y, self.car_end_heading, self.car_end_curvature)

        for point in clothoid2_car:
            clothoid_list.append(point)
            clothoid_length_list.append(point.length)
            total_length += point.length

        traj_num = int(np.ceil(total_length/self.resolution + 0.5))
        for i in range(traj_num):
            s = i * self.resolution
            if s > total_length:
                break
            else:
                if s > clothoid_length_list[0] + clothoid_length_list[1]:
                    x = clothoid_list[2].X(
                        s - clothoid_length_list[0] - clothoid_length_list[1])
                    y = clothoid_list[2].Y(
                        s - clothoid_length_list[0] - clothoid_length_list[1])
                    theta = clothoid_list[2].Theta(
                        s - clothoid_length_list[0] - clothoid_length_list[1])
                    curvature = self.car_start_curvature + \
                        clothoid_list[0].dk * clothoid_length_list[0] + \
                        clothoid_list[1].dk * clothoid_length_list[1]
                    curvature = curvature + \
                        clothoid_list[2].dk * \
                        (s - clothoid_length_list[0] - clothoid_length_list[1])
                    car_x_list.append(x)
                    car_y_list.append(y)
                    car_theta_list.append(theta)
                    car_curvature_list.append(curvature)
                    car_s_list.append(s)
                elif s > clothoid_length_list[0]:
                    x = clothoid_list[1].X(s - clothoid_length_list[0])
                    y = clothoid_list[1].Y(s - clothoid_length_list[0])
                    theta = clothoid_list[1].Theta(s - clothoid_length_list[0])
                    curvature = self.car_start_curvature + \
                        clothoid_list[0].dk * clothoid_length_list[0] + \
                        clothoid_list[1].dk * (s - clothoid_length_list[0])
                    car_x_list.append(x)
                    car_y_list.append(y)
                    car_theta_list.append(theta)
                    car_curvature_list.append(curvature)
                    car_s_list.append(s)
                else:
                    x = clothoid_list[0].X(s)
                    y = clothoid_list[0].Y(s)
                    theta = clothoid_list[0].Theta(s)
                    curvature = self.car_start_curvature + \
                        clothoid_list[0].dk * s
                    car_x_list.append(x)
                    car_y_list.append(y)
                    car_theta_list.append(theta)
                    car_curvature_list.append(curvature)
                    car_s_list.append(s)

        car_refline["x_list"] = car_x_list
        car_refline["y_list"] = car_y_list
        car_refline["theta_list"] = car_theta_list
        car_refline["curvature_list"] = car_curvature_list
        car_refline["s_list"] = car_s_list
        car_refline["total_length"] = total_length
        return car_refline


def PlotRefline(ego, car, ego_refline, car_refline, map_element_pts, sample_ego_pts_list, sample_car_pts_list, best_policy, outpath):
    ego_x_list = ego_refline["x_list"]
    ego_y_list = ego_refline["y_list"]

    car_x_list = car_refline["x_list"]
    car_y_list = car_refline["y_list"]

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    # ego and car shape
    ego_vertice = list(GetVertice(ego.id, ego.param, ego.state, 0, 0))
    agent_vertice = list(GetVertice(car.id, car.param, car.state, 0, 0))
    ego_shape_xs = [ego_vertice[0][0], ego_vertice[1][0],
                    ego_vertice[2][0], ego_vertice[3][0], ego_vertice[0][0]]
    ego_shape_ys = [ego_vertice[0][1], ego_vertice[1][1],
                    ego_vertice[2][1], ego_vertice[3][1], ego_vertice[0][1]]
    car_shape_xs = [agent_vertice[0][0], agent_vertice[1][0],
                    agent_vertice[2][0], agent_vertice[3][0], agent_vertice[0][0]]
    car_shape_ys = [agent_vertice[0][1], agent_vertice[1][1],
                    agent_vertice[2][1], agent_vertice[3][1], agent_vertice[0][1]]
    plt.plot(ego_shape_xs, ego_shape_ys)
    plt.plot(car_shape_xs, car_shape_ys)

    # plot sample pts
    sample_ego_xs_list = list()
    sample_car_xs_list = list()
    sample_ego_ys_list = list()
    sample_car_ys_list = list()
    for sample_ego_pts in sample_ego_pts_list:
        sample_xs = list()
        sample_ys = list()
        for pts in sample_ego_pts:
            sample_xs.append(pts.state.x)
            sample_ys.append(pts.state.y)
        plt.plot(sample_xs, sample_ys, linewidth=0.5, color="green")
    for sample_car_pts in sample_car_pts_list:
        sample_xs = list()
        sample_ys = list()
        for pts in sample_car_pts:
            sample_xs.append(pts.state.x)
            sample_ys.append(pts.state.y)
        plt.plot(sample_xs, sample_ys, linewidth=0.5, color="blue")

    best_xs = list()
    best_ys = list()
    for pts in best_policy:
        best_xs.append(pts.state.x)
        best_ys.append(pts.state.y)
    plt.plot(best_xs, best_ys, color="red")

    for pts in map_element_pts:
        x_list = pts[0]
        y_list = pts[1]
        plt.plot(x_list, y_list)

    plt.plot(car_y_list, car_x_list, '--')

    plt.plot(ego_y_list, ego_x_list, '--')

    plt.plot([-5, -5], [50, 20], '--')
    plt.plot([20, 50], [-5, -5], '--')

    plt.axis("equal")
    plt.savefig("{}".format(outpath))
    fig.clf()
    plt.clf()
    plt.cla()
    plt.close(fig)
    plt.close('all')
    gc.collect()


def CreateGIF(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.2)
    return


def DelJPGs(jpgs):
    for jpg in jpgs:
        try:
            os.remove(jpg)
        except BaseException as e:
            pass

# only for scenario (other: left, ego: straight)


def GenReflinePts(ego_refline, car_refline):
    ego_refline_pts = [(x, y) for y, x in zip(
        ego_refline["x_list"], ego_refline["y_list"])]
    car_refline_pts = []
    for i in range(120):
        car_refline_pts.append((-5, 50 - i*0.25))
    for i in range(len(car_refline["x_list"])):
        car_refline_pts.append(
            (car_refline["y_list"][i], car_refline["x_list"][i]))
    for i in range(120):
        car_refline_pts.append((20 + i*0.25, -5))
    return ego_refline_pts, car_refline_pts


def SearchNearestPtDist(current_pt, all_pts):
    dist_list = [math.sqrt(math.pow(pt[0] - current_pt[0], 2) +
                           math.pow(pt[1] - current_pt[1], 2)) for pt in all_pts]
    dist_list.sort()
    return dist_list[0]


class SimpleGameTheory():
    def __init__(self, ego: Vehicle, agent: Vehicle, ego_refline_pts: list, car_refline_pts: list) -> None:
        self.w_czone = 200
        self.w_szone = 20
        self.w_offroad = 100
        self.w_retrograde = 10
        self.w_goaldist = 1
        self.w_ref = 5
        self.delta_t = 0.6
        self.vision = 1.8
        self.lambda_ = 0.9

        self.belief_list = [1.0, 0.0, 0.0]
        self.delta_p = 0.5

        self.lat_safe_dist = 2.0
        self.lon_safe_dist = 5.0

        self.ego = ego
        self.agent = agent

        self.ego_refline_pts = ego_refline_pts
        self.car_refline_pts = car_refline_pts

        self.ego_all_trajectories = []
        self.car_all_trajectories = []

        self.car_traj_level_0 = []
        self.car_traj_level_1 = []
        self.car_traj_level_2 = []

        self.action_types = ["Acc", "Dec",
                             "Maintain", "Turnleft", "Turnright"]

    def CostFunction(self, ego: Vehicle, agent: Vehicle, off_road_flag: bool, retrograde_flag: bool,
                     goal_pt: tuple, refline_pts: list, search_pt: tuple) -> float:
        calc_cost_time_start = time.time()
        ego_vertice_without_safe = GetVertice(
            ego.id, ego.param, ego.state, 0, 0)
        agent_vertice_without_safe = GetVertice(
            agent.id, agent.param, agent.state, 0, 0)
        ego_vertice_with_safe = GetVertice(
            ego.id, ego.param, ego.state, self.lat_safe_dist, self.lon_safe_dist)
        agent_vertice_with_safe = GetVertice(
            agent.id, agent.param, agent.state, self.lat_safe_dist, self.lon_safe_dist)
        # czone
        czone_area = CalOverlap(ego_vertice_without_safe,
                                agent_vertice_without_safe)
        czone_ = -1 if czone_area > 0 else 0
        # szone
        szone_area = CalOverlap(ego_vertice_with_safe, agent_vertice_with_safe)
        szone_ = -1 if szone_area > 0 else 0
        # off_road
        off_road_ = -1 if off_road_flag else 0
        # retrograde
        retrograde_ = -1 if retrograde_flag else 0
        # goal
        goal_dist_ = - \
            math.sqrt(math.pow(goal_pt[0] - search_pt[0],
                      2) + math.pow(goal_pt[1] - search_pt[1], 2))
        # refline
        min_dist = SearchNearestPtDist(search_pt, refline_pts)

        cost = self.w_czone * czone_ + self.w_szone * szone_ + self.w_offroad * off_road_ + \
            self.w_retrograde * retrograde_ + self.w_goaldist * \
            goal_dist_ - self.w_ref * min_dist
        calc_cost_time_end = time.time()
        return cost

    def AccumulateCost(self, cost: float, n: int) -> float:
        return math.pow(self.lambda_, n) * cost

    def SetBelief(self, belief):
        self.belief_list = belief

    def UpdateBelief(self, real_action: tuple, level_k_action_seq: list) -> float:
        update_belief_time_start = time.time()
        k = len(level_k_action_seq)
        k_star_set = dict()
        for i in range(k):
            level_i_euclidean = Cal2DEuclideanDist(
                (level_k_action_seq[i][1].state.x, level_k_action_seq[i][1].state.y), real_action)
            k_star_set[i] = level_i_euclidean
        sorted_k_star_set = dict(
            sorted(k_star_set.items(), key=lambda x: x[1]))
        k_star = list(sorted_k_star_set.keys())[0]
        self.belief_list[k_star] += self.delta_p
        total = sum(self.belief_list)
        self.belief_list = [x / total for x in self.belief_list]
        update_belief_time_end = time.time()
        print("Update Belief Time is ",
              update_belief_time_end - update_belief_time_start)
        return self.belief_list

    def AccumulateCostWithBelief(self, cost_set: list) -> float:
        costs_with_belief = [x * y for x, y in zip(self.belief_list, cost_set)]
        return sum(costs_with_belief)

    def CondidateSampling(self, vehicle: Vehicle, vehicle_type: str) -> list:
        condidate_sample_time_start = time.time()
        root = Node(vehicle)
        pending_current_nodes = [root]
        pending_nodes_records = list()
        bfs_build(vehicle_type, self.vision, self.delta_t,
                  pending_current_nodes, self.action_types, pending_nodes_records)
        leaf_nodes = pending_nodes_records[0]
        print("leaf_nodes: ", len(leaf_nodes))
        sample_seqs = list()
        for node in leaf_nodes:
            sample_seq = [node.data]
            entry = copy.deepcopy(node)
            node_loop_time_start = time.time()
            while(entry.father is not None):
                entry = copy.deepcopy(entry.father)
                sample_seq.append(entry.data)
            reversed_sample_seq = sample_seq[::-1]
            sample_seqs.append(reversed_sample_seq)
            node_loop_time_end = time.time()
            print("Single Node Time is ",
                  node_loop_time_end - node_loop_time_start)
        condidate_sample_time_end = time.time()
        print("Condidate Sample Time is ",
              condidate_sample_time_end - condidate_sample_time_start)
        return sample_seqs

    """
    level k game theory for other decision
    """

    def TransformProcessinLevelZero(self) -> list:
        level_zero_time_start = time.time()
        all_trajectorys_other = self.car_all_trajectories
        traj_cost_dict = dict()
        for index in range(len(all_trajectorys_other)):
            costs = list()
            accumulate_cost = 0
            trajectory = all_trajectorys_other[index]
            for pt in trajectory:
                costs.append(self.CostFunction(
                    pt, self.ego, False, False, (50, -5), self.car_refline_pts, (pt.state.x, pt.state.y)))
            for i in range(len(costs)):
                accumulate_cost += self.AccumulateCost(costs[i], i)
            traj_cost_dict[index] = accumulate_cost
        sorted_traj_costs = dict(
            sorted(traj_cost_dict.items(), key=lambda x: x[1], reverse=True))
        winner_index = list(sorted_traj_costs.keys())[0]
        level_zero_time_end = time.time()
        print("Level Zero Time is ", level_zero_time_end - level_zero_time_start)
        return all_trajectorys_other[winner_index]

    def TransformProcessinLevelOne(self) -> list:
        level_one_time_start = time.time()
        all_trajectorys_ego = self.ego_all_trajectories
        traj_cost_dict_ego = dict()
        for index in range(len(all_trajectorys_ego)):
            costs_ego = list()
            accumulate_cost_ego = 0
            trajectory_ego = all_trajectorys_ego[index]
            for pt in trajectory_ego:
                costs_ego.append(self.CostFunction(
                    pt, self.agent, False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            for i in range(len(costs_ego)):
                accumulate_cost_ego += self.AccumulateCost(costs_ego[i], i)
            traj_cost_dict_ego[index] = accumulate_cost_ego
        sorted_traj_costs_ego = dict(sorted(
            traj_cost_dict_ego.items(), key=lambda x: x[1], reverse=True))
        winner_ego_index = list(sorted_traj_costs_ego.keys())[0]

        winner_ego_trajectory = all_trajectorys_ego[winner_ego_index]
        all_trajectorys_other = self.car_all_trajectories
        traj_cost_dict_other = dict()
        for index in range(len(all_trajectorys_other)):
            costs_other = list()
            accumulate_cost_other = 0
            trajectory_other = all_trajectorys_other[index]
            for time_index in range(len(trajectory_other)):
                pt = trajectory_other[time_index]
                costs_other.append(self.CostFunction(
                    pt, winner_ego_trajectory[time_index], False, False, (50, -5), self.car_refline_pts, (pt.state.x, pt.state.y)))
            for i in range(len(costs_other)):
                accumulate_cost_other += self.AccumulateCost(costs_other[i], i)
            traj_cost_dict_other[index] = accumulate_cost_other
        sorted_traj_costs_other = dict(sorted(
            traj_cost_dict_other.items(), key=lambda x: x[1], reverse=True))
        winner_other_index = list(sorted_traj_costs_other.keys())[0]
        level_one_time_end = time.time()
        print("Level One Time is ", level_one_time_end - level_one_time_start)
        return all_trajectorys_other[winner_other_index]

    def TransformProcessinLevelTwo(self) -> list:
        level_two_time_start = time.time()
        winner_other_level_zero_trajectory = self.TransformProcessinLevelZero()
        all_trajectorys_ego = self.ego_all_trajectories
        traj_cost_dict_ego = dict()
        for index in range(len(all_trajectorys_ego)):
            costs_ego = list()
            accumulate_cost_ego = 0
            trajectory_ego = all_trajectorys_ego[index]
            for time_index in range(len(trajectory_ego)):
                pt = trajectory_ego[time_index]
                costs_ego.append(self.CostFunction(
                    pt, winner_other_level_zero_trajectory[time_index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            for i in range(len(costs_ego)):
                accumulate_cost_ego += self.AccumulateCost(costs_ego[i], i)
            traj_cost_dict_ego[index] = accumulate_cost_ego
        sorted_traj_costs_ego = dict(sorted(
            traj_cost_dict_ego.items(), key=lambda x: x[1], reverse=True))
        winner_ego_index = list(sorted_traj_costs_ego.keys())[0]

        winner_ego_trajectory = all_trajectorys_ego[winner_ego_index]
        all_trajectorys_other = self.car_all_trajectories
        traj_cost_dict_other = dict()
        for index in range(len(all_trajectorys_other)):
            costs_other = list()
            accumulate_cost_other = 0
            trajectory_other = all_trajectorys_other[index]
            for time_index in range(len(trajectory_other)):
                pt = trajectory_other[time_index]
                costs_other.append(self.CostFunction(
                    pt, winner_ego_trajectory[time_index], False, False, (50, -5), self.car_refline_pts, (pt.state.x, pt.state.y)))
            for i in range(len(costs_other)):
                accumulate_cost_other += self.AccumulateCost(costs_other[i], i)
            traj_cost_dict_other[index] = accumulate_cost_other
        sorted_traj_costs_other = dict(sorted(
            traj_cost_dict_other.items(), key=lambda x: x[1], reverse=True))
        winner_other_index = list(sorted_traj_costs_other.keys())[0]
        level_two_time_end = time.time()
        print("Level Two Time is ", level_two_time_end - level_two_time_start)
        return all_trajectorys_other[winner_other_index]
    """
    ego decision (response) for other agent different strategy
    """

    def CalWinnerTrajectory(self, level_k_trajectory):
        traj_cost_dict = dict()
        for index in range(len(self.ego_all_trajectories)):
            costs = list()
            accumulate_cost = 0
            trajectory = self.ego_all_trajectories[index]
            for time_index in range(len(trajectory)):
                pt = trajectory[time_index]
                costs.append(self.CostFunction(
                    pt, level_k_trajectory[time_index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            for i in range(len(costs)):
                accumulate_cost += self.AccumulateCost(costs[i], i)
            traj_cost_dict[index] = accumulate_cost
        sorted_traj_costs = dict(sorted(
            traj_cost_dict.items(), key=lambda x: x[1], reverse=True))
        winner_index = list(sorted_traj_costs.keys())[0]
        return self.ego_all_trajectories[winner_index]

    def MakeEgoCopingStrategy(self) -> list:
        all_coping_stategies = dict()
        winner_stategy = list()

        self.ego_all_trajectories = self.CondidateSampling(self.ego, "ego")
        self.car_all_trajectories = self.CondidateSampling(self.agent, "car")

        self.car_traj_level_0 = self.TransformProcessinLevelZero()
        self.car_traj_level_1 = self.TransformProcessinLevelOne()
        self.car_traj_level_2 = self.TransformProcessinLevelTwo()

        level_zero_ego_trajectory = self.CalWinnerTrajectory(
            self.car_traj_level_0)
        level_one_ego_trajectory = self.CalWinnerTrajectory(
            self.car_traj_level_1)
        level_two_ego_trajectory = self.CalWinnerTrajectory(
            self.car_traj_level_2)

        accumulate_level_zero_ego_cost = 0
        accumulate_level_one_ego_cost = 0
        accumulate_level_two_ego_cost = 0

        accumulate_cost_zero_vs_zero = 0
        accumulate_cost_zero_vs_one = 0
        accumulate_cost_zero_vs_two = 0
        accumulate_cost_one_vs_zero = 0
        accumulate_cost_one_vs_one = 0
        accumulate_cost_one_vs_two = 0
        accumulate_cost_two_vs_zero = 0
        accumulate_cost_two_vs_one = 0
        accumulate_cost_two_vs_two = 0
        # ego vs. other
        costs_zero_vs_zero = list()
        costs_zero_vs_one = list()
        costs_zero_vs_two = list()
        costs_one_vs_zero = list()
        costs_one_vs_one = list()
        costs_one_vs_two = list()
        costs_two_vs_zero = list()
        costs_two_vs_one = list()
        costs_two_vs_two = list()

        # zero
        for index in range(len(level_zero_ego_trajectory)):
            pt = level_zero_ego_trajectory[index]
            costs_zero_vs_zero.append(self.CostFunction(
                pt, self.car_traj_level_0[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            costs_zero_vs_one.append(self.CostFunction(
                pt, self.car_traj_level_1[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            costs_zero_vs_two.append(self.CostFunction(
                pt, self.car_traj_level_2[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
        for i in range(len(costs_zero_vs_zero)):
            accumulate_cost_zero_vs_zero += self.AccumulateCost(
                costs_zero_vs_zero[i], i)
            accumulate_cost_zero_vs_one += self.AccumulateCost(
                costs_zero_vs_one[i], i)
            accumulate_cost_zero_vs_two += self.AccumulateCost(
                costs_zero_vs_two[i], i)
        accumulate_level_zero_ego_cost = self.AccumulateCostWithBelief(
            [accumulate_cost_zero_vs_zero, accumulate_cost_zero_vs_one, accumulate_cost_zero_vs_two])

        # one
        for index in range(len(level_one_ego_trajectory)):
            pt = level_one_ego_trajectory[index]
            costs_one_vs_zero.append(self.CostFunction(
                pt, self.car_traj_level_0[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            costs_one_vs_one.append(self.CostFunction(
                pt, self.car_traj_level_1[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            costs_one_vs_two.append(self.CostFunction(
                pt, self.car_traj_level_2[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
        for i in range(len(costs_one_vs_zero)):
            accumulate_cost_one_vs_zero += self.AccumulateCost(
                costs_one_vs_zero[i], i)
            accumulate_cost_one_vs_one += self.AccumulateCost(
                costs_one_vs_one[i], i)
            accumulate_cost_one_vs_two += self.AccumulateCost(
                costs_one_vs_two[i], i)
        accumulate_level_one_ego_cost = self.AccumulateCostWithBelief(
            [accumulate_cost_one_vs_zero, accumulate_cost_one_vs_one, accumulate_cost_one_vs_two])

        # two
        for index in range(len(level_two_ego_trajectory)):
            pt = level_two_ego_trajectory[index]
            costs_two_vs_zero.append(self.CostFunction(
                pt, self.car_traj_level_0[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            costs_two_vs_one.append(self.CostFunction(
                pt, self.car_traj_level_1[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
            costs_two_vs_two.append(self.CostFunction(
                pt, self.car_traj_level_2[index], False, False, (5, 50), self.ego_refline_pts, (pt.state.x, pt.state.y)))
        for i in range(len(costs_two_vs_zero)):
            accumulate_cost_two_vs_zero += self.AccumulateCost(
                costs_two_vs_zero[i], i)
            accumulate_cost_two_vs_one += self.AccumulateCost(
                costs_two_vs_one[i], i)
            accumulate_cost_two_vs_two += self.AccumulateCost(
                costs_two_vs_two[i], i)
        accumulate_level_two_ego_cost = self.AccumulateCostWithBelief(
            [accumulate_cost_two_vs_zero, accumulate_cost_two_vs_one, accumulate_cost_two_vs_two])

        all_coping_stategies[0] = accumulate_level_zero_ego_cost
        all_coping_stategies[1] = accumulate_level_one_ego_cost
        all_coping_stategies[2] = accumulate_level_two_ego_cost

        sorted_all_coping_stategies = dict(sorted(
            all_coping_stategies.items(), key=lambda x: x[1], reverse=True))
        winner_id = list(sorted_all_coping_stategies.keys())[0]

        if(winner_id == 0):
            winner_stategy = level_zero_ego_trajectory
        elif (winner_id == 1):
            winner_stategy = level_one_ego_trajectory
        elif (winner_id == 2):
            winner_stategy = level_two_ego_trajectory
        return winner_stategy

    def AllTrajectories(self):
        return self.ego_all_trajectories, self.car_all_trajectories

    def OtherTrajectory(self):
        return self.car_traj_level_0, self.car_traj_level_1, self.car_traj_level_2

    def Belief(self):
        return self.belief_list


if __name__ == '__main__':
    refline_generator = GenReflineByClothoid(20, -5, math.radians(180), 0, -5, 20, math.radians(
        90), 0, -50, 5, math.radians(0), 0, 50, 5, math.radians(0), 0)
    ego_refline = refline_generator.gen_ego_refline()
    car_refline = refline_generator.gen_car_refline()
    map_element_pts = [[[0, 0], [50, 10]], [[0, 0], [-50, -10]],
                       [[-50, -10], [0, 0]], [[50, 10], [0, 0]],
                       [[-10, -50], [10, 10]], [[10, 50], [10, 10]],
                       [[-10, -50], [-10, -10]], [[10, 50], [-10, -10]],
                       [[-10, -10], [10, 50]], [[10, 10], [10, 50]],
                       [[-10, -10], [-10, -50]], [[10, 10], [-10, -50]]
                       ]
    ego_refline_pts, car_refline_pts = GenReflinePts(ego_refline, car_refline)

    param = Param()
    ego_state = State(5, -50, 5, math.radians(90), 0, 0)
    other_state = State(-5, 25, 5, math.radians(-90), 0, 0)
    ego = Vehicle(1, ego_state, param)
    other = Vehicle(2, other_state, param)

    level_zero = list()
    level_one = list()
    level_two = list()
    x_list = list()

    out_index = 0
    count = 0
    out_path_list = list()
    u_belief = [1, 0, 0]
    while(ego.state.y < 20 and out_index < 20):
        outpath = f'/home/caros/neolix/GameTheory/gif/{out_index}.jpg'
        out_path_list.append(outpath)
        model = SimpleGameTheory(ego, other, ego_refline_pts, car_refline_pts)
        print(u_belief)
        model.SetBelief(u_belief)
        best_policy = model.MakeEgoCopingStrategy()
        ego_all_trajectories, car_all_trajectories = model.AllTrajectories()
        car_traj_level_0, car_traj_level_1, car_traj_level_2 = model.OtherTrajectory()
        beliefs = model.Belief()
        x_list.append(out_index)
        level_zero.append(beliefs[0])
        level_one.append(beliefs[1])
        level_two.append(beliefs[2])
        PlotRefline(ego, other, ego_refline, car_refline, map_element_pts,
                    ego_all_trajectories, car_all_trajectories, best_policy, outpath)
        ego = best_policy[1]
        other = car_traj_level_1[1]
        print(car_traj_level_0)
        print(car_traj_level_1)
        print(car_traj_level_2)
        print(other)
        u_belief = model.UpdateBelief((other.state.x, other.state.y), [
            car_traj_level_0, car_traj_level_1, car_traj_level_2])
        out_index += 1
    gif_name = f'/home/caros/neolix/GameTheory/gif/{count}.gif'
    CreateGIF(out_path_list, gif_name)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_list, level_zero, "red")
    ax.plot(x_list, level_two, "green")
    ax.plot(x_list, level_zero, "blue")
    ax.set_xlabel("frame")
    ax.set_ylabel("probability")
    plt.show()
