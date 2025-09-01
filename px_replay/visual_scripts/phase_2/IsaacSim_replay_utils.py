"""IsaacSim_action_replay_utils.py

功能说明:
- 支持Phase 2数据回放的一些配置和函数

"""

import logging
import os
import signal

from px_replay.config.paths import ASSET_PATH

right_encoder_split_dict = {
    "M4J": ("thumb_far_mag", 0, -360.0 / 65535),  # 0
    "M5J": ("thumb_far_mag", 1, 360.0 / 65535),  # 1
    "S4J": ("index_far_mag", 0, -360.0 / 65535),  # 2
    "S5J": ("index_far_mag", 1, 360.0 / 65535),  # 3
    "S2J": ("index_near_mag", 0, -360.0 / 65535),  # 4
    "S3J": ("index_near_mag", 1, 360.0 / 65535),  # 5
    "Z4J": ("middle_far_mag", 0, -360.0 / 65535),  # 6
    "Z5J": ("middle_far_mag", 1, 360.0 / 65535),  # 7
    "Z2J": ("middle_near_mag", 0, -360.0 / 65535),  # 8
    "Z3J": ("middle_near_mag", 1, 360.0 / 65535),  # 9
    "J1J": ("mag1", 0, -360.0 / 65535),  # 10
    "J2J": ("mag2", 0, -360.0 / 65535),  # 11 -----
    "W4J": ("ring_far_mag", 0, -360.0 / 65535),  # 12
    "W5J": ("ring_far_mag", 1, 360.0 / 65535),  # 13
    "W2J": ("ring_near_mag", 0, -360.0 / 65535),  # 14
    "W3J": ("ring_near_mag", 1, 360.0 / 65535),  # 15
    "X4J": ("pinky_far_mag", 0, -360.0 / 65535),  # 16
    "X5J": ("pinky_far_mag", 1, 360.0 / 65535),  # 17
    "X2J": ("pinky_near_mag", 0, -360.0 / 65535),  # 18
    "X3J": ("pinky_near_mag", 1, 360.0 / 65535),  # 19
    "J4J": ("palm_mag1", 0, 360.0 / 65535),  # 20
    "J3J": ("palm_mag1", 1, 360.0 / 65535),  # 21
    ###################### right #####################
    "M1J": ("palm_mag3", 0, -360.0 / 65535),  # -----
    "S1J": ("palm_mag3", 1, 360.0 / 65535),  # -----
    "W1J": ("palm_mag3", 2, 360.0 / 65535),  # -----
    "Z1J": ("palm_mag3", 3, 360.0 / 65535),
    "X1J": ("palm_mag3", 4, 360.0 / 65535),  # -----
    "M2J": ("palm_mag3", 5, -360.0 / 65535),  # -----
    "M3J": ("palm_mag3", 6, -360.0 / 65535),
}

# 右手手腕相对于二维码的齐次变换矩阵
import numpy as np

right_wrist_tf_rel_bracelet = np.array(
    [
        [1.0, 0.0, 0.0, 0.017],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.003],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def read_pose(file):
    try:
        with open(file, "r") as f:
            lines = f.readlines()

        pose = []
        for line in lines:
            vs = line.strip().split()
            assert len(vs) == 4, vs
            for v in vs:
                pose.append(float(v))

        return np.asarray(pose).reshape(4, 4)
    except:
        return


def get_best_t(data_tps, aligned_tps):
    t = 0
    tp = int(data_tps[t])
    best_t_ls = []
    for i in range(len(aligned_tps)):
        aligned_tp = aligned_tps[i]
        t_ls = [t]
        tp_ls = [tp]
        delta_ls = [abs(aligned_tp - tp)]
        while t < len(data_tps) - 1:
            t += 1
            tp = int(data_tps[t])
            t_ls.append(t)
            tp_ls.append(tp)
            # print(np - tp, vision_x2_tp, tp)
            delta_ls.append(abs(int(aligned_tp) - int(tp)))
            if tp >= aligned_tp:
                break

        min_delta = min(delta_ls)
        assert min_delta < 10, min_delta
        min_i = delta_ls.index(min_delta)
        min_t = t_ls[min_i]
        best_t_ls.append(min_t)
    # print(f"best_sensor_t_ls: {best_sensor_t_ls}")

    return best_t_ls


# 设置日志配置
def setup_logging(file_name_prefix):
    """设置日志配置"""
    # 创建日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名，包含时间戳
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{file_name_prefix}_{timestamp}.log")
    print(f"log info recorded path is :{os.path.abspath(log_file)}")

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 设置最低日志级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台显示INFO及以上级别
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG及以上级别
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 设置一些第三方库的日志级别
    logging.getLogger("isaaclab").setLevel(logging.INFO)
    logging.getLogger("omni").setLevel(logging.WARNING)
    logging.getLogger("carb").setLevel(logging.WARNING)

    return root_logger


def plot_joint_curves(
    time_stamps, commanded_joint_angles, actual_joint_angles, joint_names
):
    import matplotlib.pyplot as plt

    num_joints = len(joint_names)

    # 绘制关节角度曲线
    fig, axes = plt.subplots(
        nrows=(num_joints + 3) // 4,
        ncols=4,
        figsize=(20, 3 * ((num_joints + 3) // 4)),
    )
    axes = axes.flatten()
    for i, name in enumerate(joint_names):
        axes[i].plot(time_stamps, commanded_joint_angles[i], label="Set", color="r")
        axes[i].plot(time_stamps, actual_joint_angles[i], label="Actual", color="b")
        axes[i].set_title(name)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Angle (rad)")
        axes[i].legend()
    plt.tight_layout()
    plt.show()


def plot_hand_pose(time_stamps, set_poses, actual_poses):
    import matplotlib.pyplot as plt

    # 绘制关节角度曲线
    set_poses = np.stack(set_poses, axis=0)
    actual_poses = np.stack(actual_poses, axis=0)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    axes = axes.flatten()
    pose_labels = ["x", "y", "z", "qw", "qx", "qy", "qz"]
    for i in range(7):
        axes[i].plot(time_stamps, set_poses[:, i], label="Set", color="r")
        axes[i].plot(time_stamps, actual_poses[:, i], label="Actual", color="b")
        axes[i].set_title(f"Pose {pose_labels[i]}")
        axes[i].set_xlabel("Frame")
        axes[i].legend()
    axes[-1].axis("off")
    plt.tight_layout()
    plt.show()


# 根据外骨骼手关节进行设置--匹配usd读取顺序
HAND_JOINT_CONFIGS = {
    "right": {
        "J1J": 0.00,
        "J2J": 0.00,
        "J3J": 0.00,
        "J4J": 0.00,
        "M1J": 0.00,
        "S1J": 0.00,
        "W1J": 0.00,
        "X1J": 0.00,
        "Z1J": 0.00,
        "M2J": 0.00,
        "S2J": 0.00,
        "W2J": 0.00,
        "X2J": 0.00,
        "Z2J": 0.00,
        "M3J": 0.00,
        "S3J": 0.00,
        "W3J": 0.00,
        "X3J": 0.00,
        "Z3J": 0.00,
        "M4J": 0.00,
        "S4J": 0.00,
        "W4J": 0.00,
        "X4J": 0.00,
        "Z4J": 0.00,
        "M5J": 0.00,
        "S5J": 0.00,
        "W5J": 0.00,
        "X5J": 0.00,
        "Z5J": 0.00,
    }
}
OBJECT_MODEL_CONFIGS = {}

# 添加工作台模型配置
WORK_BENCH_MODEL_CONFIG = {
    "work_bench": {
        "usd_path": f"{ASSET_PATH}/work_bench.usd",
        "scale": np.array([1.5, 1.5, 1.5]),
        "mass": 0.0001,
        "init_pos": (-0.455, 0.005, 10.18),
        "collision_enabled": False,
        "rigid_body_enabled": True,  # 添加刚体属性
        "kinematic_enabled": False,  # 添加运动学属性
    }
}

if __name__ == "__main__":
    print(ASSET_PATH)
