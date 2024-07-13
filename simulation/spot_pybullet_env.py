import gym
from gym import spaces
import numpy as np
from simulation import walking_controller
import random
from collections import deque
import pybullet
from simulation import bullet_client
import pybullet_data
from simulation import get_terrain_normal as normal_estimator

leg_position = ["fl_", "bl_", "fr_", "br_"]
knee_constraint_point_hip = [0.0263, 0, -0.0423]  # hip
knee_constraint_point_knee = [-0.06211, 0, -0.07395]  # knee
no_of_points = 100


def constrain_theta(theta):
    """
    Lấy phần dư của phép chia theta / 2 * no_of_points
    Ràng buộc theta không vượt quá 200
    :param theta: chu kỳ
    :return: theta
    """
    theta = np.fmod(theta, 2 * no_of_points)
    if theta < 0:
        theta = theta + 2 * no_of_points
    return theta


class SpotEnv(gym.Env):
    def __init__(self,
                 render=False,
                 on_rack=False,
                 gait='trot',
                 phase=(0, no_of_points, no_of_points, 0),
                 action_dim=12,
                 end_steps=1000,
                 stairs=False,
                 downhill=False,
                 seed_value=100,
                 wedge=True,
                 imu_noise=False,
                 deg=5,
                 test=False,
                 default_pos=(-0.23, 0, 0.3)):
        """
        Class for Spotdog
        :param render: render the pybullet environment
        :param on_rack: put robot on the rack
        :param gait: dáng đi
        :param phase: pha của mỗi chân
        :param action_dim: kích thước hành động
        :param end_steps: số tập kết thúc
        :param stairs: cầu thang
        :param downhill: xuống dốc
        :param seed_value: seed value
        :param wedge: dốc
        :param imu_noise: nhiễu IMU
        :param deg:
        """

        self.new_fric_val = None
        self._motor_id_list = None
        self._joint_name_to_id = None
        self.spot = None
        self.wedge = None
        self.robot_landing_height = None
        self.wedgeOrientation = None
        self.wedgePos = None
        self.wedge_halfheight = None
        self.plane = None
        self._is_stairs = stairs
        self._is_wedge = wedge
        self._is_render = render
        self._on_rack = on_rack
        self.rh_along_normal = 0.24

        self.seed_value = seed_value
        random.seed(self.seed_value)

        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()

        self._theta = 0

        self._frequency = 2.5
        self.termination_steps = end_steps
        self.downhill = downhill

        # PD gains
        self._kp = 200
        self._kd = 10

        self.dt = 0.005
        self._frame_skip = 25
        self._n_steps = 0
        self._action_dim = action_dim

        self._obs_dim = 11

        self.action = np.zeros(self._action_dim)

        self._last_base_position = [0, 0, 0]
        self.last_yaw = 0

        self.current_com_height = 0.243

        # wedge_parameters
        self.wedge_start = 0.5
        self.wedge_halflength = 2

        self.test = test
        if gait is 'trot':
            phase = [0, no_of_points, no_of_points, 0]
        elif gait is 'walk':
            phase = [0, no_of_points, 3 * no_of_points / 2, no_of_points / 2]

        self._walkcon = walking_controller.WalkingController(gait_type=gait, phase=phase)

        self.inverse = False
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0

        self.avg_vel_per_step = 0
        self.avg_omega_per_step = 0

        self.linearV = 0
        self.angV = 0
        self.prev_vel = [0, 0, 0]

        self.x_f = 0
        self.y_f = 0

        self.clips = 7

        self.friction = 0.6
        self.ori_history_length = 3
        self.ori_history_queue = deque([0] * 3 * self.ori_history_length, maxlen=3 * self.ori_history_length)

        self.step_disp = deque([0] * 100, maxlen=100)
        self.stride = 5

        self.incline_deg = deg
        self.incline_ori = 0

        self.prev_incline_vec = (0, 0, 1)

        self.add_imu_noise = imu_noise

        self.INIT_POSITION = default_pos
        self.INIT_ORIENTATION = [0, 0, 0, 1]
        self.desired_height = 0

        self.support_plane_estimated_pitch = 0
        self.support_plane_estimated_roll = 0

        self.perturb_steps = 0

        self._obs_dim = 3 * self.ori_history_length + 2  # [r,p,y]x previous time steps, suport plane roll and pitch
        observation_high = np.array([np.pi / 2] * self._obs_dim)
        observation_low = -observation_high
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.hard_reset()

        self.set_randomization(default=True, idx1=2, idx2=2)
        self.height = []

        if self._is_stairs:
            boxhalflength = 0.1
            boxhalfwidth = 1
            boxhalfheight = 0.015
            sh_colbox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                                                                   halfExtents=[boxhalflength, boxhalfwidth,
                                                                                boxhalfheight])
            boxorigin = 0.3
            n_steps = 15
            self.stairs = []
            for i in range(n_steps):
                step = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colbox,
                                                             basePosition=[boxorigin + i * 2 * boxhalflength, 0,
                                                                           boxhalfheight + i * 2 * boxhalfheight],
                                                             baseOrientation=[0.0, 0.0, 0.0, 1])
                self.stairs.append(step)
                self._pybullet_client.changeDynamics(step, -1, lateralFriction=0.8)

    def hard_reset(self):
        """
        1) Đặt các thông số mô phỏng mà sẽ duy trì không thay đổi trong suốt quá trình thử nghiệm.
        2) Tải các tập tin URDF của mặt phẳng (plane), miếng cản (wedge)
            và robot ở trạng thái ban đầu (initial conditions).
        :return:
        """
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self._pybullet_client.setTimeStep(self.dt / self._frame_skip)
        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 1])
        self._pybullet_client.setGravity(0, 0, -9.81)

        if self._is_wedge:

            wedge_halfheight_offset = 0.01

            self.wedge_halfheight = wedge_halfheight_offset + 1.5 * np.tan(np.radians(self.incline_deg)) / 2.0
            self.wedgePos = [0, 0, self.wedge_halfheight]
            self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

            if not self.downhill:
                wedge_model_path = "simulation/Wedges/uphill/urdf/wedge_" + str(self.incline_deg) + ".urdf"

                self.INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler(
                    [np.radians(self.incline_deg) * np.sin(self.incline_ori),
                     -np.radians(self.incline_deg) * np.cos(self.incline_ori), 0])

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * abs(self.wedge_start)

                self.INIT_POSITION = [self.INIT_POSITION[0], self.INIT_POSITION[1], self.robot_landing_height]

            else:
                wedge_model_path = "simulation/Wedges/downhill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * 1.5

                self.INIT_POSITION = [0, 0, self.robot_landing_height]  # [0.5, 0.7, 0.3] #[-0.5,-0.5,0.3]

                self.INIT_ORIENTATION = [0, 0, 0, 1]

            self.wedge = self._pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)

            self.set_wedge_friction(0.7)

        model_path = "simulation/SpotDog2305/urdf/SpotDog2305.urdf"
        self.spot = self._pybullet_client.loadURDF(model_path, self.INIT_POSITION, self.INIT_ORIENTATION)

        self._joint_name_to_id, self._motor_id_list = self.build_motor_id_list()

        num_legs = 4
        for i in range(num_legs):
            self.reset_leg(i, add_constraint=True)

        if self._on_rack:
            self._pybullet_client.createConstraint(
                self.spot, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, 0.35])

        self._pybullet_client.resetBasePositionAndOrientation(self.spot, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.spot, [0, 0, 0], [0, 0, 0])

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self.set_foot_friction(self.friction)

    def reset_standing_position(self):
        """
        Đặt lại tư thế đứng
        """
        num_legs = 4
        for i in range(num_legs):
            self.reset_leg(i, add_constraint=False, standstilltorque=10)

        # Điều kiện đứng yên
        for i in range(300):
            self._pybullet_client.stepSimulation()

        for i in range(num_legs):
            self.reset_leg(i, add_constraint=False, standstilltorque=0)

    def reset(self):
        """
        Chức năng này thiết lập lại môi trường
        :note: Hàm set_randomization() được gọi trước reset()
            để ngẫu nhiên hoặc thiết lập môi trường trong điều kiện mặc định.
        :return:
        """
        self._theta = 0
        self._last_base_position = [0, 0, 0]
        self.last_yaw = 0
        self.inverse = False

        if self._is_wedge:
            self._pybullet_client.removeBody(self.wedge)

            wedge_halfheight_offset = 0.01

            self.wedge_halfheight = wedge_halfheight_offset + 1.5 * np.tan(np.radians(self.incline_deg)) / 2.0
            self.wedgePos = [0, 0, self.wedge_halfheight]
            self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

            if not self.downhill:
                wedge_model_path = "simulation/Wedges/uphill/urdf/wedge_" + str(self.incline_deg) + ".urdf"

                self.INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler(
                    [np.radians(self.incline_deg) * np.sin(self.incline_ori),
                     -np.radians(self.incline_deg) * np.cos(self.incline_ori), 0])

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * abs(self.wedge_start)

                self.INIT_POSITION = [self.INIT_POSITION[0], self.INIT_POSITION[1], self.robot_landing_height]

            else:
                wedge_model_path = "simulation/Wedges/downhill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * 1.5

                self.INIT_POSITION = [0, 0, self.robot_landing_height]  # [0.5, 0.7, 0.3] #[-0.5,-0.5,0.3]

                self.INIT_ORIENTATION = [0, 0, 0, 1]

            self.wedge = self._pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)
            self.set_wedge_friction(0.7)

        self._pybullet_client.resetBasePositionAndOrientation(self.spot, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.spot, [0, 0, 0], [0, 0, 0])
        self.reset_standing_position()

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._n_steps = 0
        return self.get_observation()

    def apply_ext_force(self, x_f, y_f, link_index=1, visulaize=False, life_time=0.01):
        """
        Hàm áp dụng lực ngoại lực lên robot
        :param x_f: ngooại lực theo hướng x
        :param y_f: ngooại lực theo hướng y
        :param link_index: chỉ số link của robot mà lực cần được áp dụng
        :param visulaize: bool, có hiển thị lực ngoại lực bằng biểu tượng mũi tên hay không
        :param life_time: thời gian tồn tại của việc hiển thị
        :return:
        """
        force_applied = [x_f, y_f, 0]
        self._pybullet_client.applyExternalForce(self.spot, link_index, forceObj=[x_f, y_f, 0], posObj=[0, 0, 0],
                                                 flags=self._pybullet_client.LINK_FRAME)
        f_mag = np.linalg.norm(np.array(force_applied))

        if visulaize and f_mag != 0.0:
            point_of_force = self._pybullet_client.getLinkState(self.spot, link_index)[0]

            lam = 1 / (2 * f_mag)
            dummy_pt = [point_of_force[0] - lam * force_applied[0],
                        point_of_force[1] - lam * force_applied[1],
                        point_of_force[2] - lam * force_applied[2]]
            self._pybullet_client.addUserDebugText(str(round(f_mag, 2)) + " N", dummy_pt, [0.13, 0.54, 0.13],
                                                   textSize=2, lifeTime=life_time)
            self._pybullet_client.addUserDebugLine(point_of_force, dummy_pt, [0, 0, 1], 3, lifeTime=life_time)

    def get_link_mass(self, link_idx):
        """
        Chức năng để lấy khối lượng của bất kỳ liên kết nào
        :param link_idx: link index
        :return: mass of the link
        """
        m = self._pybullet_client.getDynamicsInfo(self.spot, link_idx)
        return m[0]

    def set_randomization(self, default=False, idx1=0, idx2=0, idx3=1, idxc=2, idxp=0, deg=5, ori=0):
        """
        Hàm này giúp ngẫu nhiên hóa các thông số vật lý và động lực
            của môi trường để tăng cường tính ổn định của chính sách.

        Các thông số này bao gồm độ nghiêng của miếng cản, định hướng của miếng cản,
            ma sát, khối lượng của các liên kết, sức mạnh động cơ và lực ngoại lực gây nhiễu từ bên ngoài.

        :param default: Nếu đối số mặc định là True, hàm này sẽ thiết lập các thông số được
            đề cập ở trên theo cách người dùng xác định
        :param idx1:
        :param idx2:
        :param idx3:
        :param idxc: index clip
        :param idxp: index force
        :param deg:
        :param ori:
        :return:
        """
        if default:
            friction = [0.55, 0.6, 0.8]
            clip = [5.2, 6, 7, 8]
            pertub_range = [0, -60, 60, -100, 100]
            self.perturb_steps = 150
            self.x_f = 0
            self.y_f = pertub_range[idxp]
            self.incline_deg = deg + 2 * idx1
            self.incline_ori = ori + np.pi / 6 * idx2
            self.new_fric_val = friction[idx3]
            self.friction = self.set_foot_friction(self.new_fric_val)
            self.clips = clip[idxc]

        else:
            avail_deg = [5, 7, 9, 11]
            pertub_range = [0, -60, 60, -100, 100]
            self.perturb_steps = 150  # random.randint(90,200) #`Keeping fixed for now`
            self.x_f = 0
            self.y_f = pertub_range[random.randint(0, 4)]
            self.incline_deg = avail_deg[random.randint(0, 3)]
            self.incline_ori = (np.pi / 12) * random.randint(0, 6)  # resolution of 15 degree
            self.new_fric_val = np.round(np.clip(np.random.normal(0.6, 0.08), 0.55, 0.8), 2)
            self.friction = self.set_foot_friction(self.new_fric_val)
            self.clips = np.round(np.clip(np.random.normal(6.5, 0.4), 5, 8), 2)

    def randomize_only_inclines(self, default=False, idx1=0, idx2=0, deg=5, ori=0):
        """
        Hàm này chỉ ngẫu nhiên hóa độ nghiêng và định hướng của miếng cản và
            được gọi trong quá trình huấn luyện mà không sử dụng Randomization Domain.
        """
        if default:
            self.incline_deg = deg + 2 * idx1
            self.incline_ori = ori + np.pi / 6 * idx2

        else:
            avail_deg = [5, 7, 9, 11]
            self.incline_deg = avail_deg[random.randint(0, 3)]
            self.incline_ori = (np.pi / 12) * random.randint(0, 6)  # resolution of 15 degree

    @staticmethod
    def bound_y_shift(x, y):
        """
        Hàm này giới hạn sự dịch chuyển Y liên quan đến sự dịch chuyển X hiện tại
        :param x: sự dịch chuyển X tuyệt đối
        :param y: Sự dịch chuyển Y
        :return: Sự dịch chuyển Y được giới hạn
        """
        if x > 0.5619:
            if y > 1 / (0.5619 - 1) * (x - 1):
                y = 1 / (0.5619 - 1) * (x - 1)
        return y

    def get_y_x_shift(self, yx):
        """
        Hàm này giới hạn sự dịch chuyển X và Y trong không gian làm việc hình thang
        :param yx:
        :return:
        """
        y = yx[:4]
        x = yx[4:]
        for i in range(0, 4):
            y[i] = self.bound_y_shift(abs(x[i]), y[i])
            y[i] = y[i] * 0.038
            x[i] = x[i] * 0.0418
        yx = np.concatenate([y, x])
        return yx

    @staticmethod
    def transform_action(action):
        """
        Chuyển đổi các hành động được chuẩn hóa thành các lệnh đã được tỷ lệ
        :param action: 16 dimensional 1D array of predicted action values from policy in following order :
            [(step lengths of FR, FL, BR, BL), (step height of FR, FL, BR, BL),
            (X-shifts of FR, FL, BR, BL), (Y-shifts of FR, FL, BR, BL)]
        :return: các tham số hành động đã được tỷ lệ

        :note:
        Cách đặt hệ trục Descartes cho hệ thống chân trong mã nguồn
            codebase theo thứ tự này: Y trỏ lên, X trỏ về phía trước và Z sang phải.
        Trong khi trong bài báo nghiên cứu, chúng tôi tuân theo
            thứ tự này: Z trỏ lên, X trỏ về phía trước và Y sang phải.
        """
        action = np.clip(action, -1, 1)

        action[:4] = (action[:4] + 1) / 2  # Step lengths are positive always

        action[:4] = action[:4] * 0.16  # Max step length = 0.16

        action[4:8] = (action[4:8] + 1) / 2  # Step height are positive always

        action[4:8] = action[4:8] * 0.08  # Max step height = 0.08

        action[8:12] = np.clip(action[8:12], -0.035, 0.035)  # x_shift

        action[12:16] = np.clip(action[12:16], -0.015, 0.015)  # y_shift

        return action

    def get_foot_contacts(self):
        """
        Truy xuất thông tin liên lạc của chân với mặt đất hỗ trợ và bất kỳ cấu trúc đặc biệt nào (miếng cản/cầu thang).

        :return: Mảng nhị phân 8 chiều, bốn giá trị đầu tiên
            biểu thị thông tin liên lạc của chân [FR, FL, BR, BL] với mặt đất
            trong khi bốn giá trị tiếp theo là với cấu trúc đặc biệt.
        """
        foot_ids = [7, 3, 15, 11]
        foot_contact_info = np.zeros(8)

        for leg in range(4):
            contact_points_with_ground = self._pybullet_client.getContactPoints(self.plane, self.spot, -1,
                                                                                foot_ids[leg])
            if len(contact_points_with_ground) > 0:
                foot_contact_info[leg] = 1

            if self._is_wedge:
                contact_points_with_wedge = self._pybullet_client.getContactPoints(self.wedge, self.spot, -1,
                                                                                   foot_ids[leg])
                if len(contact_points_with_wedge) > 0:
                    foot_contact_info[leg + 4] = 1

            if self._is_stairs:
                for steps in self.stairs:
                    contact_points_with_stairs = self._pybullet_client.getContactPoints(steps, self.spot, -1,
                                                                                        foot_ids[leg])
                    if len(contact_points_with_stairs) > 0:
                        foot_contact_info[leg + 4] = 1

        return foot_contact_info

    def step(self, action):
        """
        Hàm để thực hiện một bước trong môi trường
        :param action: mảng các giá trị hành động
        :return:
        1. quan sát sau khi thực hiện bước
        2. phần thưởng nhận được sau bước thực hiện
        3. liệu bước có kết thúc môi trường hay không
        4. bất kỳ thông tin nào về môi trường (sẽ được thêm sau)
        """

        if self.test is False:
            action = self.transform_action(action)
        self.do_simulation(action, n_frames=self._frame_skip)
        ob = self.get_observation()
        reward, done = self._get_reward()
        return ob, reward, done, {}

    def current_velocities(self):
        """
        Trả về vận tốc tuyến tính và góc của robot
        :return:
        1. linear velocity
        2. angular velocity
        """
        current_w = self.get_base_angular_velocity()[2]
        current_v = self.get_base_linear_velocity()
        radial_v = np.sqrt(current_v[0] ** 2 + current_v[1] ** 2)
        return radial_v, current_w

    def do_simulation(self, action, n_frames):
        """
        Chuyển đổi các tham số hành động thành các lệnh động cơ tương ứng
        với sự hỗ trợ của một bộ điều khiển quỹ đạo elip
        :param action:
        :param n_frames:
        :return:
        """
        omega = 2 * no_of_points * self._frequency

        if self.test is True:
            leg_m_angle_cmd = self._walkcon.run_elliptical(self._theta, self.test)
        else:
            leg_m_angle_cmd = self._walkcon.run_elliptical_traj_spot(self._theta, action)

        self._theta = constrain_theta(omega * self.dt + self._theta)

        m_angle_cmd_ext = np.array(leg_m_angle_cmd)

        m_vel_cmd_ext = np.zeros(8)

        force_visualizing_counter = 0

        for _ in range(n_frames):
            _ = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
            self._pybullet_client.stepSimulation()

            if self.perturb_steps <= self._n_steps <= self.perturb_steps + self.stride:
                pass
                force_visualizing_counter += 1
                if force_visualizing_counter % 7 == 0:
                    self.apply_ext_force(self.x_f, self.y_f, visulaize=False, life_time=0.1)
                else:
                    self.apply_ext_force(self.x_f, self.y_f, visulaize=False)

        contact_info = self.get_foot_contacts()
        pos, ori = self.get_base_pos_and_orientation()

        rot_mat = self._pybullet_client.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat)
        rot_mat = np.reshape(rot_mat, (3, 3))

        plane_normal, self.support_plane_estimated_roll, self.support_plane_estimated_pitch = \
            normal_estimator.vector_method_stoch2(self.prev_incline_vec, contact_info, self.get_motor_angles(), rot_mat)
        self.prev_incline_vec = plane_normal

        # line_id = self._pybullet_client.addUserDebugLine([0, 0, 0], plane_normal, lineColorRGB=[1, 0, 0], lineWidth=2)
        # if 'line_id' in globals():
        #     self._pybullet_client.removeUserDebugItem(line_id)

        self._n_steps += 1

    def _termination(self, pos, orientation):
        """
        Kiểm tra các điều kiện kết thúc của môi trường
        :param pos: vị trí hiện tại của cơ sở của robot trong hệ thống thế giới
        :param orientation: hướng hiện tại của cơ sở của robot (Quaternions) trong hệ thống thế giới
        :return: trả về True nếu các điều kiện kết thúc được đáp ứng
        """
        done = False
        rpy = self._pybullet_client.getEulerFromQuaternion(orientation)

        if self._n_steps >= self.termination_steps:
            done = True
        else:
            if abs(rpy[0]) > np.radians(30):
                print('Oops, Robot about to fall sideways! Terminated')
                done = True

            if abs(rpy[1]) > np.radians(35):
                print('Oops, Robot doing wheel! Terminated')
                done = True

            if pos[2] > 0.7:
                print('Robot was too high! Terminated')
                done = True

        return done

    def _get_reward(self):
        """
        Tính toán phần thưởng đạt được bởi robot cho ổn định RPY,
            tiêu chí chiều cao thân và quãng đường di chuyển về phía trước trên độ dốc:

        :return:
        1. phần thưởng đạt được
        2. trả về True nếu môi trường kết thúc
        """
        wedge_angle = np.radians(self.incline_deg)
        robot_height_from_support_plane = 0.26
        pos, ori = self.get_base_pos_and_orientation()

        rpy_original = self._pybullet_client.getEulerFromQuaternion(ori)
        rpy = np.round(rpy_original, 4)

        # self.height.append(pos[2])
        # print(self.height)

        current_height = round(pos[2], 5)
        self.current_com_height = current_height
        standing_penalty = 10

        desired_height = (robot_height_from_support_plane / np.cos(wedge_angle) + np.tan(wedge_angle)
                          * (pos[0] * np.cos(self.incline_ori) + 0.5))

        # roll_reward = np.exp(-600 * ((rpy[0] - self.support_plane_estimated_roll) ** 2))
        # pitch_reward = np.exp(-600 * ((rpy[1] - self.support_plane_estimated_pitch) ** 2))
        # yaw_reward = np.exp(-800 * (rpy[2] ** 2))
        # height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

        roll_reward = np.abs(rpy[0] - self.support_plane_estimated_roll)
        pitch_reward = np.abs(rpy[1] - self.support_plane_estimated_pitch)
        yaw_reward = np.abs(rpy[2])

        x = pos[0]
        x_last = self._last_base_position[0]
        self._last_base_position = pos

        step_distance_x = (x - x_last)

        done = self._termination(pos, ori)
        if done:
            reward = 0
        else:
            # reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4) \
            #          + round(height_reward, 4) + 100 * round(step_distance_x, 4)

            reward_distance = 10 * step_distance_x
            penalty = roll_reward + pitch_reward + 2 * yaw_reward
            reward = reward_distance - penalty

        # Penalize for standing at same position for continuous 150 steps
        # self.step_disp.append(step_distance_x)
        #
        # if self._n_steps > 150:
        #     if sum(self.step_disp) < 0.05:
        #         reward = reward - standing_penalty

        return reward, done

    def _apply_pd_control(self, motor_commands, motor_vel_commands):
        """
        Áp dụng điều khiển PD để đạt được các lệnh vị trí động cơ mong muốn
        :param motor_commands:
        :param motor_vel_commands:
        :return: mảng các giá trị mô men xoắn đã áp dụng theo thứ tự [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        """
        qpos_act = self.get_motor_angles()
        qvel_act = self.get_motor_velocities()
        applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)

        applied_motor_torque = np.clip(np.array(applied_motor_torque), -self.clips, self.clips)
        applied_motor_torque = applied_motor_torque

        for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
            self.set_motor_torque_by_id(motor_id, motor_torque)
        return applied_motor_torque

    @staticmethod
    def add_noise(sensor_value, sd=0.04):
        """
        Thêm nhiễu cảm biến có độ lệch chuẩn do người dùng định nghĩa vào giá trị cảm biến hiện tại
        :param sensor_value:
        :param sd:
        :return:
        """
        noise = np.random.normal(0, sd, 1)
        sensor_value = sensor_value + noise[0]
        return sensor_value

    def get_observation(self):
        """
        Hàm này trả về quan sát hiện tại của môi trường cho nhiệm vụ quan tâm
        :return: [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t)
            mặt phẳng hỗ trợ ước lượng (roll, pitch)]
        """
        pos, ori = self.get_base_pos_and_orientation()
        rpy = self._pybullet_client.getEulerFromQuaternion(ori)
        rpy = np.round(rpy, 5)

        for val in rpy:
            if self.add_imu_noise:
                val = self.add_noise(val)
            self.ori_history_queue.append(val)

        obs = np.concatenate(
            (self.ori_history_queue, [self.support_plane_estimated_roll, self.support_plane_estimated_pitch])).ravel()

        return obs

    def get_motor_angles(self):
        """
        :return: Hàm này trả về các góc khớp hiện tại theo thứ tự [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]

        """
        motor_ang = [self._pybullet_client.getJointState(self.spot, motor_id)[0] for motor_id in self._motor_id_list]
        return motor_ang

    def get_motor_velocities(self):
        """
        :return: Hàm này trả về các vận tốc của
            các khớp hiện tại theo thứ tự [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        """
        motor_vel = [self._pybullet_client.getJointState(self.spot, motor_id)[1] for motor_id in self._motor_id_list]
        return motor_vel

    def get_base_pos_and_orientation(self):
        """
        :return: Hàm này trả về vị trí cơ sở của robot (X, Y, Z) và hướng (Quaternions) trong hệ thống thế giới
        """
        position, orientation = self._pybullet_client.getBasePositionAndOrientation(self.spot)
        return position, orientation

    def get_base_angular_velocity(self):
        """
        Hàm này trả về vận tốc góc của base của robot trong hệ thống thế giới
        :return: list of 3 floats
        """
        basevelocity = self._pybullet_client.getBaseVelocity(self.spot)
        return basevelocity[1]

    def get_base_linear_velocity(self):
        """
        Hàm này trả về vận tốc tuyến tính của cơ sở của robot trong hệ thống thế giới
        :return: list of 3 floats
        """
        basevelocity = self._pybullet_client.getBaseVelocity(self.spot)
        return basevelocity[0]

    def set_foot_friction(self, foot_friction):
        """
        Hàm này điều chỉnh hệ số ma sát của chân robot
        :param foot_friction: hệ số ma sát mong muốn của chân
        :return: hệ số ma sát hiện tại
        """
        foot_link_id = [3, 7, 11, 15]
        for link_id in foot_link_id:
            self._pybullet_client.changeDynamics(self.spot, link_id, lateralFriction=foot_friction)
        return foot_friction

    def set_wedge_friction(self, friction):
        """
        Hàm này điều chỉnh hệ số ma sát của miếng cản
        :param friction: hệ số ma sát mong muốn của miếng cản
        :return:
        """
        self._pybullet_client.changeDynamics(self.wedge, -1, lateralFriction=friction)

    def set_motor_torque_by_id(self, motor_id, torque):
        """
        Hàm để đặt mô men xoắn động cơ cho motor_id tương ứng
        :param motor_id: index of motor whose torque
        :param torque: torque of motor
        :return:
        """
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.spot,
            jointIndex=motor_id,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            force=torque)

    def build_motor_id_list(self):
        """
        Hàm để ánh xạ tên khớp với motor_id tương ứng và tạo danh sách motor_ids
        :return:
        1. Từ điển từ tên khớp sang motor_id
        2. Danh sách các id của khớp tương ứng cho các động cơ
        """
        num_joints = self._pybullet_client.getNumJoints(self.spot)
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.spot, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

        motor_names = ["motor_fl_upper_hip_joint",
                       "motor_fl_upper_knee_joint",
                       "motor_fr_upper_hip_joint",
                       "motor_fr_upper_knee_joint",
                       "motor_bl_upper_hip_joint",
                       "motor_bl_upper_knee_joint",
                       "motor_br_upper_hip_joint",
                       "motor_br_upper_knee_joint"]

        motor_id_list = [joint_name_to_id[motor_name] for motor_name in motor_names]

        return joint_name_to_id, motor_id_list

    def reset_leg(self, leg_id, add_constraint, standstilltorque=10):
        """
        Hàm để thiết lập lại trạng thái của các khớp hông và đầu gối.
        :param leg_id: chỉ số của chân
        :param add_constraint: boolean để tạo ra ràng buộc trong các khớp dưới của cơ chế chân có năm thanh
        :param standstilltorque: giá trị của mô men xoắn ban đầu được đặt
            trong động cơ hông và đầu gối cho điều kiện đứng
        :return:
        """
        leg = leg_position[leg_id]
        self._pybullet_client.resetJointState(
            self.spot,
            self._joint_name_to_id["motor_" + leg + "upper_knee_joint"],  # motor
            targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.spot,
            self._joint_name_to_id[leg + "lower_knee_joint"],
            targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.spot,
            self._joint_name_to_id["motor_" + leg + "upper_hip_joint"],  # motor
            targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(
            self.spot,
            self._joint_name_to_id[leg + "lower_hip_joint"],
            targetValue=0, targetVelocity=0)

        if add_constraint:
            c = self._pybullet_client.createConstraint(
                self.spot, self._joint_name_to_id[leg + "lower_hip_joint"],
                self.spot, self._joint_name_to_id[leg + "lower_knee_joint"],
                self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
                knee_constraint_point_hip, knee_constraint_point_knee)

            self._pybullet_client.changeConstraint(c, maxForce=200)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.spot,
            jointIndex=(self._joint_name_to_id["motor_" + leg + "upper_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=standstilltorque)
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.spot,
            jointIndex=(self._joint_name_to_id["motor_" + leg + "upper_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=standstilltorque)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.spot,
            jointIndex=(self._joint_name_to_id[leg + "lower_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0)

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.spot,
            jointIndex=(self._joint_name_to_id[leg + "lower_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0)

    @property
    def pybullet_client(self):
        return self._pybullet_client
