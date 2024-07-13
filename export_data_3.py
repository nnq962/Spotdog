import simulation.arbitrary_slopes_env as e
from fabulous.color import blue, green, red, bold
import argparse
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
import simulation.spot_pybullet_env as spot
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='22.04')
    parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=1.6)
    parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=0)
    parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
    parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
    parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
    parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
    parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=5000)
    parser.add_argument('--PerturbForce',
                        help='perturbation force to applied perpendicular to the heading direction of the robot',
                        type=float, default=0.0)
    parser.add_argument('--Downhill', help='should robot walk downhill?', type=bool, default=False)
    parser.add_argument('--Stairs', help='test on staircase', type=bool, default=False)
    parser.add_argument('--AddImuNoise', help='flag to add noise in IMU readings', type=bool, default=False)
    parser.add_argument('--Test', help='flag to test with out data', type=bool, default=False)

    args = parser.parse_args()
    policy = np.load("japan_data/" + args.PolicyDir + "/iterations/best_policy.npy")

    WedgePresent = True

    if args.WedgeIncline == 0 or args.Stairs:
        WedgePresent = False
    elif args.WedgeIncline < 0:
        args.WedgeIncline = -1 * args.WedgeIncline
        args.Downhill = True

    env = spot.SpotEnv(render=True,
                       wedge=WedgePresent,
                       stairs=args.Stairs,
                       downhill=args.Downhill,
                       seed_value=args.seed,
                       on_rack=False,
                       gait='trot',
                       imu_noise=args.AddImuNoise,
                       test=args.Test,
                       default_pos=(0, 0, 0.30))

    if args.RandomTest:
        env.set_randomization(default=False)
    else:
        env.incline_deg = args.WedgeIncline
        env.incline_ori = np.radians(args.WedgeOrientation)
        env.set_foot_friction(args.FrictionCoeff)
        env.clips = args.MotorStrength
        env.perturb_steps = 300
        env.y_f = args.PerturbForce
    state = env.reset()

    if args.Test:
        print(bold(blue("\nTest with out data\n")))

    print(
        bold(blue("\nTest Parameters:\n")),
        green('\nWedge Inclination:'), red(env.incline_deg),
        green('\nWedge Orientation:'), red(np.degrees(env.incline_ori)),
        # green('\nCoeff. of friction:'), red(env.friction),
        # green('\nMotor saturation torque:'), red(env.clips)
    )

    # ================================================================
    # Khởi tạo một deque để lưu trữ giá trị của biến theo thời gian
    maxlen = 20
    data = deque(maxlen=maxlen)

    # Khởi tạo figure và axes cho đồ thị
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, maxlen - 1)
    ax.set_ylim(-0.12, 0.12)  # Thay 100 bằng giới hạn tùy ý cho trục y

    # Simulation starts
    t_r = 0

    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    count = 0

    # Hàm cập nhật dữ liệu trên đồ thị
    def update(frame):
        global state
        global count
        action = policy.dot(state)
        state, r, _, angle = env.step(action)
        # env.pybullet_client.resetDebugVisualizerCamera(1, 50, -10, env.get_base_pos_and_orientation()[0])

        pos, ori = env.get_base_pos_and_orientation()
        rpy = env._pybullet_client.getEulerFromQuaternion(ori)

        # if 560 <= count <= 1450:
        #     env.incline_deg = 0

        # Thêm giá trị mới vào deque
        data.append(env.transform_action(action)[0])  # Thay np.random.randint bằng giá trị thực tế của biến
        data_1.append(np.degrees(env.get_motor_angles()[0]))
        data_2.append(np.degrees(env.get_motor_angles()[1]))

        # if 650 <= count <= 1250:
        #     data_1.append(pos[2])
        #     data_2.append(env.desired_height)
        #     # data_3.append(env.transform_action(action)[6])
        #     # data_4.append(env.transform_action(action)[7])
        #
        if count == 999:
            print("-------------------")
            print(data_1)
            print(data_2)
        #     # print(data_3)
        #     # print(data_4)
        #     print("-------------------")

        # Cập nhật dữ liệu của đường cong
        line.set_data(range(len(data)), data)

        count += 1
        return line,


    def animate(i):
        return update(i)

    # Chạy animation
    ani = FuncAnimation(fig, animate, interval=5, blit=True)
    plt.grid()
    # Hiển thị đồ thị
    plt.show()
