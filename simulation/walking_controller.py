# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from collections import namedtuple
from utils import spot_kinematic
import numpy as np

no_of_points = 100


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


@dataclass
class LegData:
    name: str
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0


@dataclass
class RobotData:
    front_right: float
    front_left: float
    back_right: float
    back_left: float


class WalkingController:
    def __init__(self,
                 gait_type='trot',
                 phase=None,
                 ):
        if phase is None:
            phase = [0, 0, 0, 0]
        self._phase = RobotData(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = LegData('fl')
        self.front_right = LegData('fr')
        self.back_left = LegData('bl')
        self.back_right = LegData('br')
        self.gait_type = gait_type

        self.MOTOROFFSETS_Spot = [np.radians(160.2), np.radians(43.28)]

        self.Spot_kinematics = spot_kinematic.SpotKinematics()

    def update_leg_theta(self, theta):
        """
        Tùy thuộc vào dáng đi, theta cho mỗi chân sẽ được tính toán
        :param theta:
        :return:
        """
        self.front_right.theta = constrain_theta(theta + self._phase.front_right)
        self.front_left.theta = constrain_theta(theta + self._phase.front_left)
        self.back_right.theta = constrain_theta(theta + self._phase.back_right)
        self.back_left.theta = constrain_theta(theta + self._phase.back_left)

    def initialize_elipse_shift(self, x_shift, y_shift):
        self.front_right.x_shift = x_shift[0]
        self.front_left.x_shift = x_shift[1]
        self.back_right.x_shift = x_shift[2]
        self.back_left.x_shift = x_shift[3]

        self.front_right.y_shift = y_shift[0]
        self.front_left.y_shift = y_shift[1]
        self.back_right.y_shift = y_shift[2]
        self.back_left.y_shift = y_shift[3]

    def initialize_leg_state(self, theta, action):
        """
        Khởi tạo tất cả các tham số của các quỹ đạo chân

        :param theta: tham số chu kỳ quỹ đạo theta
        :param action: các tham số điều biến quỹ đạo được dự đoán bởi chính sách
        :return: namedtuple('legs', 'front_right front_left back_right back_left')
        """
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)

        self.update_leg_theta(theta)

        leg_sl = action[:4]  # fr fl br bl

        self._update_leg_step_length_val(leg_sl)

        self.initialize_elipse_shift(action[4:8], action[8:12])

        return legs

    def _update_leg_phi_val(self, leg_phi):
        """

        :param leg_phi: steering angles for each leg trajectories
        :return:
        """
        self.front_right.phi = leg_phi[0]
        self.front_left.phi = leg_phi[1]
        self.back_right.phi = leg_phi[2]
        self.back_left.phi = leg_phi[3]

    def _update_leg_step_length_val(self, step_length):
        """

        :param step_length: step length of each leg trajectories
        :return:
        """
        self.front_right.step_length = step_length[0]
        self.front_left.step_length = step_length[1]
        self.back_right.step_length = step_length[2]
        self.back_left.step_length = step_length[3]

    def run_elliptical_traj_spot(self, theta, action):
        """
        Bộ điều khiển quỹ đạo bán-ellipse

        :param theta: tham số chu kỳ quỹ đạo theta
        :param action: các tham số điều chỉnh quỹ đạo được dự đoán bởi policy
        :return: danh sách vị trí của động cơ cho hành động mong muốn
        """
        legs = self.initialize_leg_state(theta, action)

        y_center = -0.26
        foot_clearance = 0.05
        x_center = 0.01
        x = y = 0

        # step_length = 0.1
        # radius = step_length / 2
        # theta = 0
        # y_center = -0.26329
        # foot_clearance = 0.06
        # x_shift = 0.0345
        # y_shift = 0

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * np.pi
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + x_center  # + leg.x_shift
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center  # + leg.y_shift

            leg.x, leg.y = x, y

            leg.motor_hip, leg.motor_knee, _ = self.Spot_kinematics.inverse_kinematics(leg.x, leg.y, 0)

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Spot[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Spot[1]

            # leg.theta = (leg.theta / (2 * no_of_points)) * 2 * np.pi
            # x = -radius * np.cos(leg.theta) + x_shift
            # if leg.theta > np.pi:
            #     flag = 0
            # else:
            #     flag = 1
            # y = foot_clearance * np.sin(leg.theta) * flag + y_center + y_shift
            # theta += 2.5
            # theta = np.fmod(theta, 2 * no_of_points)
            #
            # leg.motor_hip, leg.motor_knee, _ = self.Spot_kinematics.inverse_kinematics(x, y, 0)
            # leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Spot[0]
            # leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Spot[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee]

        return leg_motor_angles
