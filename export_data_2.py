import simulation.spot_pybullet_env as spot
import argparse
from fabulous.color import blue, green, red, bold
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='24.04')
    parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=1.6)
    parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=9)
    parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
    parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
    parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
    parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
    parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=7000)
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
                       default_pos=(-1.2, 0, 0.32))

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
        green('\nCoeff. of friction:'), red(env.friction),
        green('\nMotor saturation torque:'), red(env.clips))

    # Simulation starts

    # Khởi tạo deque để lưu trữ dữ liệu với kích thước tối đa là 20 cho mỗi giá trị
    data = [deque(maxlen=10) for _ in range(4)]

    # Khởi tạo figure và axes FR, FL, BR, BL
    fig, ax = plt.subplots()
    labels = ['Front Right', 'Front Left', 'Back Right', 'Back Left']  # Nhãn cho các đường thẳng
    lines = [ax.plot([], [], label=labels[i])[0] for i in range(4)]  # 4 đường thẳng trên đồ thị

    # Giới hạn trục y trong khoảng từ -0.01 đến 0.13
    ax.set_ylim(-0.01, 0.14)

    step_length_1 = []
    step_length_2 = []
    step_length_3 = []
    step_length_4 = []
    count = 0

    # Tạo hàm cập nhật đồ thị
    def update_plot(frame):
        global state
        global count
        counter = 0

        for i in range(4):
            # print(count)

            # Thêm giá trị mới vào deque tương ứng
            action = policy.dot(state)
            state, r, _, angle = env.step(action)
            data[i].append(env.transform_action(action)[i])

            # if count == 2000:
            #     print("-------------------")
            #     print(step_length_1)
            #     print(step_length_2)
            #     print(step_length_3)
            #     print(step_length_4)
            #     print("-------------------")
            #
            # if counter == 0:
            #     step_length_1.append(env.transform_action(action)[i])
            # if counter == 1:
            #     step_length_2.append(env.transform_action(action)[i])
            # if counter == 2:
            #     step_length_3.append(env.transform_action(action)[i])
            # if counter == 3:
            #     step_length_4.append(env.transform_action(action)[i])
            #
            # counter += 1

            # if i % 100 == 0:
            env.pybullet_client.resetDebugVisualizerCamera(1.5, 40, -10, env.get_base_pos_and_orientation()[0])
            lines[i].set_data(range(len(data[i])), data[i])  # Cập nhật dữ liệu cho đường thẳng thứ i
            ax.set_xlim(max(0, len(data[i]) - 20), len(data[i]))  # Giới hạn trục x trong 20 bước thời gian
            # count += 1

        return lines


    # Cài đặt animation
    ani = FuncAnimation(fig, update_plot, interval=1)  # Cập nhật mỗi 50ms

    plt.grid(which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
    plt.title('Step length real-time')
    plt.xlabel('Time')
    plt.ylabel('Step length (meters)')
    ax.legend()  # Hiển thị chú thích

    plt.show()
