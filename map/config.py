import numpy as np
from .cubic_spline_planner import CubicSpline2D
import math
# import matplotlib.pyplot as plt

P_ex = np.array([-26.44, 14.31]) # 入口车道车道线中点
P_en = np.array([14.74, 43.67]) # 出口车道车道线中点
P0 = np.array([-4.39, 14.54])# 入口车道停车线中点
P3 = np.array([14.85, 35.40])# 出口车道停车线中点

def construct_line(x1, y1, x2, y2, x=None, y=None):
    # 计算两点之间的斜率
    if x1 == x2:
        k = None  # 竖直方向上的直线，斜率不存在
    else:
        k = (y2 - y1) / (x2 - x1)

    # 计算直线的截距
    b = y1 - k * x1 if k is not None else None

    if k is None:  # 竖直方向上的直线
        assert x is not None, "竖直方向上的直线必须给定 x 坐标"
        return x, y1
    else:
        assert x is None or y is None, "只能指定 x 或 y 中的一个"
        if x is not None:
            y = k * x + b
        else:
            x = (y - b) / k
        return x, y

def generate_target_course(x, y):
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rs, rx, ry, ryaw, rk = [], [], [], [], []
    for i_s in s:
        rs.append(i_s)
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    csp.s = rs

    return rx, ry, ryaw, rk, csp

def LeftRefLineBuild():
    ##-------------------------左转车参考路径----------------------
    P1_x, P1_y = construct_line(P0[0], P0[1], P_ex[0], P_ex[1], x=P0[0] + 5 * math.cos(46.4 / 360 * math.pi))
    P1 = np.array([P1_x, P1_y])
    P2_x, P2_y = construct_line(P3[0], P3[1], P_en[0], P_en[1], y=P3[1] - 5 * math.cos(46.4 / 360 * math.pi))
    P2 = np.array([P2_x, P2_y])
    t = np.linspace(0, 1, 100)
    B = np.zeros((100, 2))
    for i in range(100):
        B[i] = (1 - t[i]) ** 3 * P0 + 3 * t[i] * (1 - t[i]) ** 2 * P1 + 3 * t[i] ** 2 * (1 - t[i]) * P2 + t[i] ** 3 * P3

    reference_line = np.vstack((P_ex, B))
    reference_line = np.vstack((reference_line, P_en))

    wx = reference_line[:, 0]
    wy = reference_line[:, 1]
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    return tx, ty, tyaw, csp

def StrRefLineBuild():
    # 构造参考线
    P_ex = np.array([100, 18.12])
    P_en = np.array([-26, 18.12])

    reference_line = np.vstack((P_ex, P_en))

    wx = reference_line[:, 0]
    wy = reference_line[:, 1]
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    return tx, ty, tyaw, csp

global tx, ty, tyaw, csp
tx, ty, tyaw, csp = LeftRefLineBuild()
# plt.plot(tx, ty)
# plt.plot(P_ex[0], P_ex[1], "xr")
# plt.plot(P_en[0], P_en[1], "xr")
# plt.plot(P0[0], P0[1], "or")
# plt.plot(P3[0], P3[1], "ob")
# plt.show()
