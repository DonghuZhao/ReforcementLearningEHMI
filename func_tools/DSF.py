import numpy as np


def _calc_DSF(controlled_vehicle, other_vehicle):
    """
    Calculate the DSF value between two vehicles
    :param controlled_vehicle: ControlledVehicle
    :param other_vehicle: Vehicle
    :return: DSF value
    """
    G = 0.001
    k1 = 1
    k2 = 0.05
    M1 = 4000 # kg
    M2 = 4000 # kg
    v1 = controlled_vehicle.speed
    v2 = other_vehicle.speed
    position1 = controlled_vehicle.position
    position2 = other_vehicle.position
    heading1 = controlled_vehicle.heading
    heading2 = other_vehicle.heading
    d_squared = (position1[0] - position2[0])**2 + (position1[1] - position2[1])**2
    cos_theta = np.cos(heading1 - heading2)
    return -G * M1 * M2 * np.exp( k2 * (v1 + v2)) / d_squared * cos_theta

