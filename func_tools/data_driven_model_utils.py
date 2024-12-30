import math
import numpy as np
from utils.trajectory import State
from math import sin, cos, radians, sqrt
from shapely.geometry import Polygon, Point


class DataDrivenUtils:
    """用于数据驱动模型的工具类,提供状态转换、观测计算等功能"""
    
    def __init__(self):
        self.MAX_SPEED = 30  # metadrive中的限速
        self.MAX_ACC = 5  # 车辆加速度阈值
        
    @staticmethod
    def radians_to_degrees(radians):
        """将角度从弧度单位转换为度单位。

        Args:
        - radians : float
            角度，以弧度为单位。

        Returns:
        - degrees : float
            角度，以度为单位。
            """
        degrees = radians * (180 / math.pi)
        return degrees

    @staticmethod
    def action_to_cartesian(x, y, v, acc, yaw, steering, dt):
        """
        更新车辆状态。

        参数:
        x (float): 当前 x 坐标
        y (float): 当前 y 坐标
        v (float): 当前速度
        yaw (float): 当前航向角（弧度）
        acc (float): 当前加速度
        steer_rate (float): 转向角速度（每秒转向角变化量，单位：弧度/秒）
        dt (float): 时间步长
        L (float): 车辆轴距

        返回:
        (float, float, float, float): 更新后的 (x, y, v, yaw)
        """
        # 计算此时刻的转向角变化
        delta_steer = steering * dt

        # 更新速度
        v_new = v + acc * dt

        # 更新位置
        x_new = x + v * math.cos(yaw) * dt
        y_new = y + v * math.sin(yaw) * dt

        # 更新航向角
        yaw_new = yaw + (v / 3) * math.tan(delta_steer) * dt

        return x_new, y_new, v_new, yaw_new

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        """计算两点之间的距离。

        Args:
            x1, y1, x2, y2: 两点全局坐标

        Returns:
            绝对距离
        """
        return np.float32(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    @staticmethod
    def calculate_min_distance(center1, heading1, center2, heading2, length=5, width=3):
        """计算两车几何图形之间的最小距离
        """
        # 辅助函数，计算矩形四个角的坐标
        def get_corners(center, heading, length, width):
            dx = length / 2 * cos(heading)
            dy = length / 2 * sin(heading)
            dx_perp = width / 2 * cos(heading + np.pi / 2)
            dy_perp = width / 2 * sin(heading + np.pi / 2)
            corners = [
                (center[0] - dx + dx_perp, center[1] - dy + dy_perp),
                (center[0] - dx - dx_perp, center[1] - dy - dy_perp),
                (center[0] + dx - dx_perp, center[1] + dy - dy_perp),
                (center[0] + dx + dx_perp, center[1] + dy + dy_perp)
            ]
            return corners

        # 获取两车的角坐标
        corners1 = get_corners(center1, heading1, length, width)
        corners2 = get_corners(center2, heading2, length, width)

        # 创建多边形
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)

        # 计算两个多边形之间的最小距离
        min_distance = poly1.distance(poly2)

        return min_distance


    def adjust_v_yaw(yaw_rad):
        """
        转换车辆的航向角坐标系,使其保持在逆时针为[-360,0]
                -270/90
                    |
        -180/180 ——   —— 0
                    |
                -90
        """
        # 针对小于-360度的情况
        if yaw_rad < -2 * math.pi:
            yaw_rad = yaw_rad + 2 * math.pi

        # 针对大于0的情况
        if yaw_rad > 0:
            yaw_rad = -2 * math.pi + yaw_rad

        # 判断yaw_rad是否在[-2pi,0]区间，如果不是则唤起错误
        if yaw_rad < -2 * math.pi or yaw_rad > 0:
            raise ValueError("yaw_rad is not in the range of [-2pi, 0]")

        return yaw_rad

    @staticmethod
    def calculate_angle(main_car_pos, main_car_heading, other_car_pos):
        """
        计算其他车辆相对于主车前进方向的角度。

        参数:
        main_car_pos (tuple): 主车的位置 (x, y)
        main_car_heading (float): 主车的航向角度，以度为单位
        other_car_pos (tuple): 其他车辆的位置 (x, y)

        返回:
        float: 其他车辆相对于主车前进方向的角度（度）
        """
        # 将主车航向角从[-2π, 0]转换到[0, 2π]
        main_car_heading = - main_car_heading

        # 计算向量AB
        vector_AB = ([other_car_pos[0] - main_car_pos[0], other_car_pos[1] - main_car_pos[1]])

        # 东向向量
        vector_east = (1, 0)

        # 计算点积和叉积
        dot = vector_east[0] * vector_AB[0] + vector_east[1] * vector_AB[1]
        cross = - (vector_AB[1] * vector_east[0] - vector_AB[0] * vector_east[1])

        # 计算角度
        angle = np.arctan2(cross, dot)
        angle_degrees = np.degrees(angle)

        # 调整角度到[0, 360)
        if angle_degrees < 0:
            angle_degrees += 360

        ego_yaw_degrees = DataDrivenUtils.radians_to_degrees(main_car_heading)
        angle_diff = angle_degrees - ego_yaw_degrees

        if angle_diff < 0:
            angle_diff += 360

        return angle_diff



    def update_lidar_with_closest_cars(distances, angles,n_interval: int = 36):
        """
        更新lidar列表，根据车辆的距离和角度信息，并只保留每个区间内距离最近的车辆。

        参数:
        distances (list): 每辆车相对于主车的距离列表。
        angles (list): 每辆车相对于主车的角度列表。

        返回:
        list: 更新后的lidar列表。
        """
        # 初始化lidar列表，所有值设为1000
        lidar = [1000] * n_interval
        interval = 360 / n_interval
        # 计算每辆车所属的区间，并将距离最近的车辆ID放入对应的区间
        for car_id, (distance, angle) in enumerate(zip(distances, angles), start=1):
            # 计算对应的区间索引
            index = int(angle // interval)
            # 由于区间是从0到35，我们需要将360度归类到第0个元素
            if index == n_interval:
                index = 0
            # 确保index在范围内
            if index >= 0 and index < n_interval:
                current_car_dist = lidar[index]
            else:
                print("Index out of range:", index)
                index = n_interval - 1
                current_car_dist = lidar[index]
                # continue
            # print('angle:', angle)
            if current_car_dist == 1000 or distance < distances[current_car_dist - 1]:
                lidar[index] = car_id

        # 将车辆ID替换为对应的距离值
        for i in range(len(lidar)):
            if lidar[i] != 1000:  # 说明有车辆id
                # 获取车辆ID对应的距离
                lidar[i] = distances[lidar[i] - 1]  # car_id是从1开始的，而索引是从0开始的

        return lidar


    def process_radar_information(self,vehicles):
        """处理并返回所有车辆的雷达信息。

        Args:
            vehicles (dict): 车辆字典,键为车辆ID,值为车辆状态对象。

        Returns:
            dict: 每个车辆ID对应的雷达观测数据。
        """
        all_v_ids = list(vehicles.keys())
        num_agents = len(all_v_ids)

        glob_x = []
        glob_y = []
        glob_v = []
        glob_yaw = []

        # 提取所有车辆的位置坐标
        for agent_id in all_v_ids:
            x = vehicles[agent_id].current_state.x
            y = vehicles[agent_id].current_state.y
            yaw = vehicles[agent_id].current_state.yaw
            yaw = DataDrivenUtils.adjust_v_yaw(yaw)
            # print('rad:', yaw)
            # print('degree:', radians_to_degrees(yaw))
            v = vehicles[agent_id].current_state.vel
            glob_x.append(np.float32(x))
            glob_y.append(np.float32(y))
            glob_v.append(np.float32(v))
            glob_yaw.append(np.float32(yaw))

        glob_x = np.array(glob_x)
        glob_y = np.array(glob_y)
        glob_v = np.array(glob_v)
        glob_yaw = np.array(glob_yaw)

        #  最终返回的雷达信息和冲突信息
        lidar_obs = {}
        ttcp = {}

        # 计算每个车辆与其他车辆的距离
        for agent_id in range(num_agents):
            agent_position = (glob_x[agent_id], glob_y[agent_id])
            distances = []
            angles = []
            allv_ttcp = []

            for vv in range(num_agents):
                if vv != agent_id:
                    other_position = (glob_x[vv], glob_y[vv])
                    distance = DataDrivenUtils.calculate_distance(agent_position[0], agent_position[1],
                                                other_position[0], other_position[1])
                    distances.append(distance)

                    # angles
                    angle = DataDrivenUtils.calculate_angle(agent_position, glob_yaw[agent_id], other_position)
                    angles.append(angle)

                    #  min ttc
                    v_ttcp = DataDrivenUtils.calculate_ttc_pet(agent_position[0], agent_position[1],
                                            glob_v[agent_id], glob_yaw[agent_id],
                                            other_position[0], other_position[1],
                                            glob_v[vv], glob_yaw[vv], distance)
                    allv_ttcp.append(v_ttcp)

            # 固定信息到对应角度
            lidars = DataDrivenUtils.update_lidar_with_closest_cars(distances, angles)

            # 归一化距离
            max_distance = np.float32(30.0)

            norm_lidar = [min(lidar / max_distance, 1.0) for lidar in lidars]
            agent_lidar1 = np.array(norm_lidar, dtype=np.float32)
            lidar_obs[all_v_ids[agent_id]] = agent_lidar1
            # 返回最小的ttcp
            ttcp[all_v_ids[agent_id]] = min(allv_ttcp) if allv_ttcp else float('inf')

        return lidar_obs, ttcp


    def calculate_ttc_pet(egox, egoy, egov, egoyaw_rad, otherx, othery, otherv, otheryaw_rad, distance):

        # 计算本车的速度分量
        egovx = egov * math.cos(egoyaw_rad)
        egovy = egov * math.sin(egoyaw_rad)
        # 计算其他车辆的速度分量
        othervx = otherv * math.cos(otheryaw_rad)
        othervy = otherv * math.sin(otheryaw_rad)
        egoyaw = DataDrivenUtils.radians_to_degrees(egoyaw_rad)
        otheryaw = DataDrivenUtils.radians_to_degrees(otheryaw_rad)
        # 计算航向角差的绝对值
        angle_diff = abs(egoyaw - otheryaw) % 360
        # 角度差可能超过180度，取最小的角度差
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        # 计算距离
        dx = otherx - egox
        dy = othery - egoy
        delta_vx = egovx - othervx
        delta_vy = egovy - othervy
        # 判断冲突对象是否在主车行进方向的后方
        dot_product = dx * egovx + dy * egovy  # 向量点积
        if dot_product < 0:  # 如果点积小于0，表示冲突对象在主车后方
            return float('inf') 

        if angle_diff > 10:
        # 两车角度大于30度，计算 PET,交叉冲突
            if (egovx == 0 and egovy == 0) or (othervx == 0 and othervy == 0) or (delta_vx == 0 and delta_vy == 0):
                return float('inf')  # 如果任一车辆不动，则没有冲突，PET 为无穷大
            else:
                # 初始化到冲突点的时间
                ttc_ego = float('inf')
                ttc_other = float('inf')
                
                # 计算主车到冲突点的时间
                if delta_vx != 0:
                    ttc_ego = dx / delta_vx
                if delta_vy != 0:
                    ttc_ego = min(ttc_ego, dy / delta_vy)
                
                # 计算其他车到冲突点的时间
                if delta_vx != 0:
                    ttc_other = -dx / othervx if othervx != 0 else float('inf')
                if delta_vy != 0:
                    ttc_other = min(ttc_other, -dy / othervy if othervy != 0 else float('inf'))
                
                # 计算 PET
                pet = ttc_ego - ttc_other
                if pet > 0:
                    return pet
                else:
                    # 对于主车更快到达冲突点
                    return float('inf')
        else:
        # 计算 TTC
            rel_vx = egovx - othervx
            rel_vy = egovy - othervy
            if abs(rel_vx) < 0.3 and abs(rel_vy) < 0.3:
                ttc = float('inf')  # 无碰撞
            else:
                relative_speed = math.sqrt(rel_vx ** 2 + rel_vy ** 2)
                ttc = distance / relative_speed
            return ttc  # 返回 None 表示不计算 PET


    def action_2_xy(self,action_q, veh, current_lane, path,
                    engine_force: float = 1.1, dt: float = 0.3, predict_states: bool = False):
        """ 处理数据驱动模型输出的动作为可被sumo执行的运动状态。

        Args:
            action: 数据驱动模型输出的动作[steer, acc]
            veh: 车辆对象。
            current_lane: 当前车辆的车道
            path: 用于存储车辆运动状态的字典
            engine_force: 用于调整车辆行为符合物理仿真引擎效果
            dt: 动作执行的单步时间
            predict_states: 是否需要计算多帧轨迹

        Returns:
            path: 更新后的存储车辆运动状态的字典
            """

        raw_steer = action_q['steer'][0]
        raw_acc = action_q['acc'][0]

        if raw_acc > 0:
            acc = min(raw_acc, 1)
        elif raw_acc == 0:
            acc = 0
        else:
            acc = max(raw_acc, -1)
        acc = 1 * acc  # 调整车辆行为符合物理仿真引擎效果

        if raw_steer > 0:  # 左转
            raw_steer = min(raw_steer, 1)
        elif raw_steer == 0:
            raw_steer = 0
        else:  # 右转
            raw_steer = max(raw_steer, -1)
        steer = 7 * raw_steer

        # 将动作转换为轨迹,输入为上一帧的车辆状态
        x, y, v, yaw = DataDrivenUtils.action_to_cartesian(x=veh.current_state.x,
                                        y=veh.current_state.y,
                                        v=veh.current_state.vel,
                                        acc=acc,
                                        yaw=veh.current_state.yaw,
                                        steering=steer,
                                        dt=dt,
                                        )
        # 计算车辆frenet坐标系
        course_spline = current_lane.course_spline
        s, d = course_spline.cartesian_to_frenet1D(x, y)

        # 把轨迹和转向角赋值给轨迹对象
        if predict_states:
            # 最低预测时长5s,预测频率0.1s，与原生轨迹生成器一致
            for t in np.arange(0, 0.5, 0.1):
                path.states.append(State(t=t,
                                        x=x, y=y, vel=v, yaw=yaw, acc=acc, s=s, d=d,
                                        laneID=current_lane.id,
                                        ))
                # 以当前状态模拟下一帧状态
                x, y, v, yaw_new = DataDrivenUtils.action_to_cartesian(x=x, y=y, v=v, acc=acc, yaw=yaw, steering=steer, dt=dt, )
                # yaw = yaw * 0.1
                s, d = course_spline.cartesian_to_frenet1D(x, y)
        else:
            # 只把当前帧的状态给轨迹对象
            path.states.append(State(x=x, y=y, vel=v, yaw=yaw, acc=acc, s=s, d=d,
                                    laneID=current_lane.id,
                                    ))

        return path


    def keep_state_trajectory(self, veh, path, current_lane, predict_states: bool = True):
        """
        构建一个轨迹对象,使车辆状态与上一帧保持不变。
        
        Args:
            veh (Vehicle): 车辆对象。
            path (Trajectory): 已存在的轨迹对象。
            predict_states (bool): 是否预测未来状态。如果为True,将生成多个相同的状态;否则,只生成当前状态。
        
        Returns:
            Trajectory: 更新后的轨迹对象,其中车辆状态保持不变。
        """
        if predict_states:
            # 最低预测时长1秒，预测频率0.1秒，与action_2_xy一致
            for t in np.arange(0, 1, 0.1):
                path.states.append(State(
                    t=t,
                    x=veh.current_state.x,
                    y=veh.current_state.y,
                    vel=veh.current_state.vel,
                    yaw=veh.current_state.yaw,
                    acc=veh.current_state.acc,
                    s=veh.current_state.s,
                    d=veh.current_state.d,
                    laneID=current_lane.id
                ))
        else:
            # 只添加当前状态
            path.states.append(State(
                t=0,  # 当前时间步
                x=veh.current_state.x,
                y=veh.current_state.y,
                vel=0,
                yaw=veh.current_state.yaw,
                acc=0,
                s=veh.current_state.s,
                d=veh.current_state.d,
                laneID=current_lane.id
            ))
        return path


    def extract_v_hist(self, vehicles_info):
        """
        从车辆信息中提取每辆车的倒数第二个历史偏航角。

        Args:
        - vehicles_info : dict
            包含车辆信息，其中 'carInAoI' 键对应一个列表，列表中的每个元素是一个包含至少两个键（'id' 和 'yawQ'）的字典。

        Returns:
        - v_hist : dict
            键为车辆ID，值为该车辆的倒数第二个历史偏航角。如果历史数据不足，则值为 None。
        """
        # 初始化字典以存储结果
        v_hist = {}

        # 遍历车辆历史记录中的所有车辆
        for vehicle in vehicles_info.get("carInAoI", []):  # 使用 get 方法避免 KeyError
            vehicle_id = vehicle.get('id')
            yaw_history = vehicle.get('yawQ', [])  # 使用 get 方法避免 KeyError

            # 检查 'yawQ' 列表是否有足够的元素
            if len(yaw_history) >= 2:
                last_second_value = yaw_history[-2]  # 获取倒数第二个元素
            else:
                last_second_value = None  # 如果没有足够的元素，设置为None

            # 将键值对添加到字典中
            v_hist[vehicle_id] = last_second_value

        return v_hist


    def calculate_lane_yaw(self, lane_spline, x_vehicle, y_vehicle):
        """根据车道中心线坐标和车辆当前位置，计算对应车道中心线的角度（朝向）。

        Args:
        - lane_spline : list of tuple
            车道中心线的坐标点列表，每个元素是一个包含(x, y)坐标的元组。
        - x_vehicle : float
            车辆当前的全局x坐标。
        - y_vehicle : float
            车辆当前的全局y坐标。

        Returns:
        - lane_yaw : float
            车道中心线在最接近车辆的点的方向角度，单位为度（degree）。
        """

        # 初始化最小距离为无穷大，用于寻找最接近车辆坐标的点
        min_distance = float('inf')
        closest_index = None

        # 遍历所有点，找到与车辆坐标最接近的点
        for i, (x, y) in enumerate(lane_spline):
            distance = math.sqrt((x - x_vehicle) ** 2 + (y - y_vehicle) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        # 确定最接近的线段
        if closest_index is None:
            return 0

        if closest_index == 0:
            # 如果最接近的点是第一个点，使用第一条线段
            point1 = lane_spline[0]
            point2 = lane_spline[1]
        elif closest_index == len(lane_spline) - 1:
            # 如果最接近的点是最后一个点，使用最后一条线段
            point1 = lane_spline[-2]
            point2 = lane_spline[-1]
        else:
            # 通常情况，选择最接近点的前后点以求更好的平滑性
            point1 = lane_spline[closest_index - 1]
            point2 = lane_spline[closest_index + 1]

        # 计算两点间的斜率
        dy = point2[1] - point1[1]
        dx = point2[0] - point1[0]

        # 使用math.atan2得到角度（弧度制）
        angle_radians = math.atan2(dy, dx)

        # 将弧度转换为度
        lane_yaw = math.degrees(angle_radians)

        return lane_yaw


    def clip(value, min_value, max_value):
        return max(min_value, min(value, max_value))


    def get_marl_obs(self, veh, roadgraph, current_lane, v_hist, lidar_obs, is_test: bool = False):
        """
        提取多智能体强化学习模型所需要的观测状态，包括主车状态，导航状态，雷达点云三部分。

        Ego states: [
                        [距道路左右边界的归一化距离,
                        ],

                        行驶方向与车道方向的角度差,
                        速度归一化值,
                        上一时间步的加速度归一化值,
                        归一化的当前角速度,

                        [当前车道中心的归一化偏移量],
                        ], dim >= 7
            Navi info: [
                        车道终点在车辆前进方向（x轴）上的相对位置,
                        当前车道的弯曲半径（曲率的倒数，+逆时针）,
                        当前车道的弯曲方向，+1顺时针，-1逆时针,
                    ] , dim = 3

            Lidar points: [
                            从主车射线起点到最近的9辆车的距离与射线总长度的比例
                            ], dim = 9

        Args:
            veh (dict)：{current_state, lane_id, id}。
            roadgraph (dict)：表示道路网络的数据结构，包括车道信息和交叉口。
            current_lane (dict)：代表车辆当前所在车道的对象。
            v_hist (dict)：车辆状态的历史记录。
            lidar_obs (list)：来自LIDAR的传感器数据，用于检测车辆周围的对象。
            is_test (bool)：标志用于确定是否用于测试新模型。

        Returns:
            states(list)，车辆当前状态, dim = 7 + 3 + 9 = 19
        """
        #  初始化状态对象
        states = []
        # PART 1 主车状态
        #  dim 1 & 2 距道路边界左、右的归一化距离[0,1]na
        d = veh.current_state.d
        left_lane_width = 0
        right_lane_width = 0

        #  判断当前车道是否处在交叉口
        if veh.lane_id in roadgraph.lanes:
            #  对于不在交叉口的车辆，获取相邻车道并计算宽度
            w = roadgraph.lanes[veh.lane_id].width
            # 获取左侧车道
            if current_lane.left_lane():
                left_lane = roadgraph.get_lane_by_id(current_lane.left_lane())
                left_lane_width = left_lane.width
                # 获取右侧车道
            if current_lane.right_lane():
                right_lane = roadgraph.get_lane_by_id(current_lane.right_lane())
                right_lane_width = right_lane.width
        # 对于交叉口的连接段，适当扩大可行车道边界，增加决策空间，不计算相邻车道（因为交叉口属于面域无法获取）
        elif veh.lane_id in roadgraph.junction_lanes:
            w = 2 * roadgraph.junction_lanes[veh.lane_id].width
        else:
            w = 4
            print(f'-------------{veh.lane_id}车道信息缺失--------------')

        # 计算最大道路边界
        max_d2l = w / 2 + left_lane_width
        max_d2r = w / 2 + right_lane_width
        total_width = w + left_lane_width + right_lane_width
        dl = max_d2l - d  # dist to left
        dr = max_d2r + d  # dist to right
        clip_dl = DataDrivenUtils.clip(dl / total_width, 0, 1.0)
        clip_dr = DataDrivenUtils.clip(dr / total_width, 0, 1.0)
        states.append(clip_dl)
        states.append(clip_dr)

        # dim 3 行驶方向与车道角度差
        v_yaw = DataDrivenUtils.radians_to_degrees(veh.current_state.yaw)
        # 车辆在当前车道上的位置
        x = veh.current_state.x
        y = veh.current_state.y
        # 获取车道中心线坐标
        lane_spline = current_lane.center_line
        lane_yaw = self.calculate_lane_yaw(lane_spline, x, y)

        # 将坐标系由[-180, 180]转换为[-360, 0]
        yaw_diff = lane_yaw - v_yaw
        if yaw_diff > 90:
            lane_yaw = lane_yaw - 360
        elif yaw_diff < - 90:
            v_yaw = v_yaw - 360
        yaw_diff = lane_yaw - v_yaw
        clip_yaw_diff = DataDrivenUtils.clip(yaw_diff, -1, 1) / 2 + 0.5
        states.append(clip_yaw_diff)

        # dim 4 速度归一化值
        MAX_SPEED = 30  # metadrive中的限速
        clip_v = DataDrivenUtils.clip((veh.current_state.vel + 1) / (MAX_SPEED + 1), 0, 1.0)
        states.append(clip_v)

        #  dim 5 上一时间步加速度归一化值
        MAX_ACC = 5  # 车辆加速度阈值
        clip_a = DataDrivenUtils.clip(abs(veh.current_state.acc + MAX_ACC) / 2 * MAX_ACC, 0, 1.0)
        states.append(clip_a)
        
        # dim 6 角速度 angular acceleration (yaw rate)
        last_yaw = v_hist[veh.id]
        current_yaw = veh.current_state.yaw
        if last_yaw:
            yaw_diff = last_yaw - current_yaw
        else:
            yaw_diff = 0
        yaw_rate = abs(DataDrivenUtils.radians_to_degrees(yaw_diff) / 0.1)
        clip_yaw_rate = DataDrivenUtils.clip(yaw_rate / 150, 0, 1)
        states.append(clip_yaw_rate)

        #  dim 7 当前车道中心的归一化偏移量
        clip_d = DataDrivenUtils.clip((d / w + 1) / 2, 0, 1.0)
        states.append(clip_d)

        # ——————————————————————————————————————————————————————————————————————————————————————————————————————————
        # PART 2 导航信息
        # dim 8 checkpoint在车辆前进方向（x轴）上的相对位置，归一化映射到[0,1]。
        # TODO: limsim存在车辆经过车道终点后，s更新不及时，导致出现负数的情况
        s = abs(veh.current_state.s)

        #  判断车道属于路段还是连接段，分别属于两个字典
        if veh.lane_id in roadgraph.lanes:
            length = roadgraph.lanes[veh.lane_id].sumo_length
        elif veh.lane_id in roadgraph.junction_lanes:
            length = roadgraph.junction_lanes[veh.lane_id].sumo_length
        else:
            length = 30
            print(f'-------------{veh.lane_id}车道信息缺失--------------')
        lead = DataDrivenUtils.clip((length - s) / length, 0, 1)
        states.append(lead)

        # dim 9 当前车道的弯曲半径，归一化映射到[0,1]
        cur = veh.current_state.cur
        bending_radius = 1 / cur if cur != 0 else 0
        states.append(bending_radius)

        # dim 10 当前车道的弯曲方向，+1表示顺时针，-1表示逆时针，归一化映射到[0,1]。
        clockwise = -1 if cur >= 0 else 1  # 曲率为+是左转，即逆时针，对应弯曲方向是-1
        states.append(clockwise)
        # ——————————————————————————————————————————————————————————————————————————————————————————————————————————
        # PART 3 雷达点云
        # dim 11-19 加入雷达模块
        # 从前面计算的雷达信息中获得当前车辆的观测信息
        lobs = lidar_obs[veh.id].tolist()  # 格式转换
        states = states + lobs
        # PART 4 社交倾向
        svo = 0.6 #[-1, 1],越大越利己
        states.append(svo)
        return states
