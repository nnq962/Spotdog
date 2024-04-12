""" Test Kinematics simulation """
from dataclasses import dataclass
from collections import namedtuple
from utils import spot_kinematic
import numpy as np

no_of_points = 100


@dataclass
class LegData:
    name: str
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    b: float = 1.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 0.0


@dataclass
class RobotData:
    front_right: float
    front_left: float
    back_right: float
    back_left: float


def constrain_theta(theta):
    """
    Lấy phần dư của phép chia theta / 2 * no_of_points

    :param theta:
    :return:
    """
    theta = np.fmod(theta, 2 * no_of_points)
    if theta < 0:
        theta = theta + 2 * no_of_points
    return theta


class TestWalking:
    def __init__(self,
                 gait_type='trot',
                 phase=(0, 0, 0, 0)):
        self._phase = RobotData(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = LegData('fl')
        self.front_right = LegData('fr')
        self.back_left = LegData('bl')
        self.back_right = LegData('br')
        self.gait_type = gait_type

        self.MOTOROFFSETS_Spot = [np.radians(152.4), np.radians(20)]
        self.Spot_kinematics = spot_kinematic.SpotKinematics()

    def update_leg_theta(self, theta):
        self.front_right.theta = constrain_theta(theta + self._phase.front_right)
        self.front_left.theta = constrain_theta(theta + self._phase.front_left)
        self.back_right.theta = constrain_theta(theta + self._phase.back_right)
        self.back_left.theta = constrain_theta(theta + self._phase.back_left)

    def initialize_leg_state(self, theta):
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)
        self.update_leg_theta(theta)
        return legs

    def run_elliptical_traj_spot(self, theta):
        legs = self.initialize_leg_state(theta)

        step_length = 0.10
        foot_clearance = 0.05

        radius = step_length / 2
        y_center = -0.22082
        x_start = 0.00811 + radius
        x = y = None

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * np.pi

            if self.gait_type == "trot":
                x = -radius * np.cos(leg_theta) + x_start
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center

            leg.x, leg.y = x, y

            leg.motor_hip, leg.motor_knee, _ = self.Spot_kinematics.inverse_kinematics(leg.x, leg.y, 0)

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Spot[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Spot[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee]

        return leg_motor_angles
