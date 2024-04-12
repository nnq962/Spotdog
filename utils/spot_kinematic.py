""" Kinematic for SpotDog - NNQ """
import numpy as np


class Serial2RKin:
    def __init__(self,
                 base_pivot=(0, 0),
                 link_lengths=(0.3, 0.3)):
        self.link_lengths = link_lengths
        self.base_pivot = base_pivot

    def inverse_kinematics(self, ee_pos, branch=1):
        q = np.zeros(2, float)
        x_y_points = np.array(ee_pos) - np.array(self.base_pivot)
        [x, y] = x_y_points.tolist()
        q1_temp = None
        [l1, l2] = self.link_lengths

        # Check if the end-effector point lies in the workspace of the manipulator
        if ((x ** 2 + y ** 2) > (l1 + l2) ** 2) or ((x ** 2 + y ** 2) < (l1 - l2) ** 2):
            print("Point is outside the workspace")
            valid = False
            return valid, q

        a = 2 * l2 * x
        b = 2 * l2 * y
        c = l1 ** 2 - l2 ** 2 - x ** 2 - y ** 2

        if branch == 1:
            q1_temp = np.arctan2(y, x) + np.arccos(-c / np.sqrt(a ** 2 + b ** 2))
        elif branch == 2:
            q1_temp = np.arctan2(y, x) - np.arccos(-c / np.sqrt(a ** 2 + b ** 2))

        q[0] = np.arctan2(y - l2 * np.sin(q1_temp), x - l2 * np.cos(q1_temp))
        q[1] = q1_temp - q[0]
        valid = True

        return valid, q

    def jacobian(self, q):
        """
        Provides the Jacobian matrix for the end-effector
        Args:
        --- q : The joint angles of the manipulator [q_hip, q_knee]
        where the angle q_knee is specified relative to the thigh link
        Returns:
        --- mat : A 2x2 velocity Jacobian matrix of the manipulator
        """
        [l1, l2] = self.link_lengths
        mat = np.zeros([2, 2])
        mat[0, 0] = -l1 * np.sin(q[0]) - l2 * np.sin(q[0] + q[1])
        mat[0, 1] = - l2 * np.sin(q[0] + q[1])
        mat[1, 0] = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        mat[1, 1] = l2 * np.cos(q[0] + q[1])
        return mat


class SpotKinematics:
    """
    SpotKinematics class by NNQ
    """
    def __init__(self,
                 base_pivot1=(0, 0),
                 base_pivot2=(0.05, 0),
                 link_parameters=(0.11, 0.25, 0.11, 0.2)):
        self.base_pivot1 = base_pivot1
        self.base_pivot2 = base_pivot2
        self.link_parameters = link_parameters

    def inverse2d(self, ee_pos):
        """
        2D inverse kinematics
        :param ee_pos: end_effector position
        :return:
        """
        valid = False
        q = np.zeros(4)
        [l1, l2, l3, l4] = self.link_parameters
        [l, _] = self.base_pivot2

        leg1 = Serial2RKin(self.base_pivot1, [l1, l2])
        leg2 = Serial2RKin(self.base_pivot2, [l3, l4])

        valid1, q1 = leg1.inverse_kinematics(ee_pos, branch=1)
        if not valid1:
            return valid, q

        ee_pos_new = [ee_pos[0] - l * np.cos(q1[0] + q1[1]), ee_pos[1] - l * np.sin(q1[0] + q1[1])]
        valid2, q2 = leg2.inverse_kinematics(ee_pos_new, branch=2)
        if not valid2:
            return valid, q

        valid = True
        q = [q1[0], q2[0], q1[0] + q1[1], q2[0] + q2[1]]
        return valid, q

    def inverse_kinematics(self, x, y, z):
        """
        Spot's inverse kinematics
        :param x: x position
        :param y: y position
        :param z: z position
        :return:
        """
        motor_abduction = np.arctan2(z, -y)
        _, [motor_hip, motor_knee, _, _] = self.inverse2d([x, y])

        if motor_hip > 0:
            motor_hip = -2 * np.pi + motor_hip

        return [motor_hip, motor_knee, motor_abduction]

    def forward_kinematics(self, q):
        """
        Spot's forward kinematics
        :param q: [hip_angle, knee_angle]
        :return: end-effector position
        """
        [l1, _, _, l2] = self.link_parameters
        [l, _] = self.base_pivot2

        a = (l1 * np.cos(q[0]) - l1 * np.cos(q[1]) - l) / l2
        b = (l1 * np.sin(q[0]) - l1 * np.sin(q[1])) / l2

        theta2 = -2 * np.arctan((2 * b + (-(a ** 2 + b ** 2) * (a ** 2 + b ** 2 - 4))
                                 ** 0.5) / (a ** 2 - 2 * a + b ** 2))

        x = l1 * np.cos(q[0]) + l2 * np.cos(theta2)
        y = l1 * np.sin(q[0]) + l2 * np.sin(theta2)

        x = x + l * np.cos(theta2)
        y = y + l * np.sin(theta2)

        ee_pos = [x, y]

        vaild = True
        return vaild, ee_pos
