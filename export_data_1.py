import simulation.arbitrary_slopes_env as e
import argparse
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='24.04')
parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=10000)
args = parser.parse_args()

policy = np.load("japan_data/" + args.PolicyDir + "/iterations/best_policy.npy")

env = e.SpotEnv(render=True, on_rack=False, gait='trot', default_pos=(0, 0, 0.30))

t_r = 0
state = env.reset()

# Khởi tạo deque để lưu trữ dữ liệu với kích thước tối đa là 20 cho mỗi giá trị
data = [deque(maxlen=40) for _ in range(4)]

# Khởi tạo figure và axes
fig, ax = plt.subplots()
lines = [ax.plot([], [])[0] for _ in range(4)]  # 4 đường thẳng trên đồ thị

# Giới hạn trục y trong khoảng từ -0.01 đến 0.13
ax.set_ylim(-0.01, 0.14)


# Tạo hàm cập nhật đồ thị
def update_plot(frame):
    global state
    for i in range(4):
        # Thêm giá trị mới vào deque tương ứng
        action = policy.dot(state)
        state, r, _, angle = env.step(action)
        data[i].append(env.transform_action(action)[i])
        lines[i].set_data(range(len(data[i])), data[i])  # Cập nhật dữ liệu cho đường thẳng thứ i
        ax.set_xlim(max(0, len(data[i]) - 20), len(data[i]))  # Giới hạn trục x trong 20 bước thời gian
    return lines


# Cài đặt animation
ani = FuncAnimation(fig, update_plot, interval=10)  # Cập nhật mỗi 50ms

plt.grid()
plt.title('Step length real-time')
plt.xlabel('Time')
plt.ylabel('Step length (meters)')

plt.show()

print("Total_reward " + str(t_r))
