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
    step_height: float = 0.0
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
                 phase=(0, 0, 0, 0)
                 ):
        self._phase = RobotData(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = LegData('fl')
        self.front_right = LegData('fr')
        self.back_left = LegData('bl')
        self.back_right = LegData('br')
        self.gait_type = gait_type

        self.MOTOROFFSETS_Spot = [np.radians(160.2), np.radians(43.28)]

        self.Spot_kinematics = spot_kinematic.SpotKinematics()

        self.step_length_1 = []
        self.step_length_2 = []
        self.step_length_3 = []
        self.step_length_4 = []

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

    def initialize_leg_state(self, theta, action, test=False):
        """
        Khởi tạo tất cả các tham số của các quỹ đạo chân

        :param theta: tham số chu kỳ quỹ đạo theta
        :param action: các tham số điều biến quỹ đạo được dự đoán bởi chính sách
        :param test: test IK
        :return: namedtuple('legs', 'front_right front_left back_right back_left')
        """
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)

        self.update_leg_theta(theta)

        if test is False:
            leg_sl = action[:4]  # fr fl br bl
            leg_sh = action[4:8]

            self._update_leg_step_length_val(leg_sl)
            self._update_leg_step_height_val(leg_sh)
            self.initialize_elipse_shift(action[8:12], action[12:16])

        return legs

    def _update_leg_step_length_val(self, step_length):
        """

        :param step_length: step length of each leg trajectories
        :return:
        """
        self.front_right.step_length = step_length[0]
        self.front_left.step_length = step_length[1]
        self.back_right.step_length = step_length[2]
        self.back_left.step_length = step_length[3]

    def _update_leg_step_height_val(self, step_height):
        """

        :param step_height: step length of each leg trajectories
        :return:
        """
        self.front_right.step_height = step_height[0]
        self.front_left.step_height = step_height[1]
        self.back_right.step_height = step_height[2]
        self.back_left.step_height = step_height[3]

    def run_elliptical_traj_spot(self, theta, action):
        """
        Bộ điều khiển quỹ đạo bán-ellipse

        :param theta: tham số chu kỳ quỹ đạo theta
        :param action: các tham số điều chỉnh quỹ đạo được dự đoán bởi policy
        :return: danh sách vị trí của động cơ cho hành động mong muốn
        """
        legs = self.initialize_leg_state(theta, action)

        x_center = 0.02
        y_center = -0.29
        x = y = 0

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * np.pi
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + x_center + leg.x_shift
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = leg.step_height * np.sin(leg_theta) * flag + y_center + leg.y_shift

            leg.x, leg.y = x, y

            leg.motor_hip, leg.motor_knee, _ = self.Spot_kinematics.inverse_kinematics(leg.x, leg.y, 0)

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Spot[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Spot[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee]

        return leg_motor_angles

    def run_elliptical(self, theta, test):
        """
        Run elipse trajectory with IK
        """
        legs = self.initialize_leg_state(theta, action=None, test=test)

        # Parameters for elip --------------------
        step_length = 0.10
        step_height = 0.05
        x_center = 0.02
        y_center = -0.29
        # ----------------------------------------

        x = y = 0
        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * np.pi
            leg.r = step_length / 2
            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + x_center
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = step_height * np.sin(leg_theta) * flag + y_center

            leg.x, leg.y = x, y
            leg.motor_hip, leg.motor_knee, _ = self.Spot_kinematics.inverse_kinematics(leg.x, leg.y, 0)

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Spot[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Spot[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee]

        return leg_motor_angles
