import os
import pandas as pd
import matplotlib.pyplot as plt

# 切换到脚本所在的目录，确保读取文件
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取 CSV 文件（确保你的文件名是 "joint_states_data.csv"）
df = pd.read_csv("joint_states_data.csv")

print(df['name'].head())
# 转换列数据为数值型，强制转换时如果不能转换则变为 NaN
# df['time'] = pd.to_numeric(df['time'], errors='coerce')
# df['position'] = pd.to_numeric(df['position'], errors='coerce')
# df['velocity'] = pd.to_numeric(df['velocity'], errors='coerce')
# df['acceleration'] = pd.to_numeric(df['acceleration'], errors='coerce')

# 如果 time 列有空值，则需要将其转换为秒
# df['time'] = df['time'] / 1e9  # 转换为秒

# 绘图函数
def plot_joint_data(joint_data):
#     # 筛选指定关节的数据
    # joint_data = df[df['joint_name'] == joint_name]
    
    # if joint_data.empty:
    #     print(f"No data found for joint: {joint_name}")
    #     return
    
#     # 将数据列转换为 NumPy 数组，以便避免多维索引问题
    time_sec = joint_data['timestamp'].to_numpy()  # 转换为 NumPy 数组
    time_nane_sec = joint_data['time'].to_numpy()*1e-9
    time = time_sec +time_nane_sec
    position = joint_data['position'].to_numpy()  # 转换为 NumPy 数组
    velocity = joint_data['velocity'].to_numpy()  # 转换为 NumPy 数组
    # acceleration = joint_data['acceleration'].to_numpy()  # 转换为 NumPy 数组
    
#     # 创建子图
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

#     # 位置图
    ax[0].plot(time, position, label='Position', color='b', marker='o')
    ax[0].set_title('Position')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (m)')
    ax[0].grid(True)

#     # 速度图
    ax[1].plot(time, velocity, label='Velocity', color='r')
    ax[1].set_title('Velocity')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[1].grid(True)

#     # 加速度图
    # ax[2].plot(time, acceleration, label='Acceleration', color='g')
    # ax[2].set_title('Acceleration')
    # ax[2].set_xlabel('Time (s)')
    # ax[2].set_ylabel('Acceleration (m/s²)')
    # ax[2].grid(True)

#     # 显示图形
    plt.tight_layout()
    plt.show()

# # 调用函数绘制 'ur5_1_wrist_3_joint' 的数据
plot_joint_data(df)
