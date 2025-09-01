import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import argparse


def load_pose_matrices(folder_path):
    """加载所有位姿矩阵文件"""
    pose_files = []
    pose_matrices = []

    # 获取所有txt文件并按数字顺序排序
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    txt_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    for file_path in txt_files:
        try:
            matrix = np.loadtxt(file_path)
            if matrix.shape == (4, 4):
                pose_files.append(file_path)
                pose_matrices.append(matrix)
            else:
                print(f"警告: {file_path} 不是4x4矩阵，跳过")
        except Exception as e:
            print(f"错误: 无法加载 {file_path}: {e}")

    return pose_files, np.array(pose_matrices)


def extract_rotation_translation(poses):
    """从位姿矩阵中提取旋转和平移"""
    rotations = []
    translations = []

    for pose in poses:
        # 提取旋转矩阵 (3x3)
        R_matrix = pose[:3, :3]
        # 提取平移向量 (3x1)
        t_vector = pose[:3, 3]

        rotations.append(R_matrix)
        translations.append(t_vector)

    return np.array(rotations), np.array(translations)


def smooth_rotations_exponential(rotations, alpha=0.3):
    """使用指数移动平均平滑旋转（四元数）"""
    quaternions = []
    for rot_matrix in rotations:
        try:
            r = R.from_matrix(rot_matrix)
            quaternions.append(r.as_quat())
        except Exception as e:
            print(f"警告: 旋转矩阵转换失败: {e}")
            quaternions.append(np.array([0, 0, 0, 1]))

    quaternions = np.array(quaternions)
    smoothed_quaternions = [quaternions[0]]  # 第一个不变

    for i in range(1, len(quaternions)):
        prev_quat = smoothed_quaternions[-1]
        curr_quat = quaternions[i]

        # 确保四元数在同一半球（避免长路径插值）
        if np.dot(prev_quat, curr_quat) < 0:
            curr_quat = -curr_quat

        # 指数移动平均
        smoothed_quat = (1 - alpha) * prev_quat + alpha * curr_quat
        smoothed_quat = smoothed_quat / np.linalg.norm(smoothed_quat)  # 归一化
        smoothed_quaternions.append(smoothed_quat)

    # 转换回旋转矩阵
    smoothed_rotations = []
    for quat in smoothed_quaternions:
        r = R.from_quat(quat)
        smoothed_rotations.append(r.as_matrix())

    return np.array(smoothed_rotations)


def smooth_rotations_gaussian(rotations, sigma=2.0):
    """使用高斯滤波平滑旋转（四元数）"""
    quaternions = []
    for rot_matrix in rotations:
        try:
            r = R.from_matrix(rot_matrix)
            quaternions.append(r.as_quat())
        except Exception as e:
            print(f"警告: 旋转矩阵转换失败: {e}")
            quaternions.append(np.array([0, 0, 0, 1]))

    quaternions = np.array(quaternions)

    # 确保四元数连续性
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i - 1], quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]

    # 对每个四元数分量应用高斯滤波
    smoothed_quaternions = np.zeros_like(quaternions)
    for i in range(4):
        smoothed_quaternions[:, i] = gaussian_filter1d(quaternions[:, i], sigma=sigma)

    # 重新归一化
    for i in range(len(smoothed_quaternions)):
        smoothed_quaternions[i] = smoothed_quaternions[i] / np.linalg.norm(smoothed_quaternions[i])

    # 转换回旋转矩阵
    smoothed_rotations = []
    for quat in smoothed_quaternions:
        r = R.from_quat(quat)
        smoothed_rotations.append(r.as_matrix())

    return np.array(smoothed_rotations)


def smooth_rotations_savgol(rotations, window_length=15, polyorder=3):
    """使用Savitzky-Golay滤波平滑旋转"""
    quaternions = []
    for rot_matrix in rotations:
        try:
            r = R.from_matrix(rot_matrix)
            quaternions.append(r.as_quat())
        except Exception as e:
            print(f"警告: 旋转矩阵转换失败: {e}")
            quaternions.append(np.array([0, 0, 0, 1]))

    quaternions = np.array(quaternions)

    # 确保四元数连续性
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i - 1], quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]

    # 确保窗口长度不超过数据长度且为奇数
    if window_length >= len(quaternions):
        window_length = len(quaternions) - 1 if len(quaternions) % 2 == 0 else len(quaternions) - 2
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        window_length = 3

    # 对每个四元数分量应用Savitzky-Golay滤波
    smoothed_quaternions = np.zeros_like(quaternions)
    for i in range(4):
        smoothed_quaternions[:, i] = savgol_filter(quaternions[:, i], window_length, polyorder)

    # 重新归一化
    for i in range(len(smoothed_quaternions)):
        smoothed_quaternions[i] = smoothed_quaternions[i] / np.linalg.norm(smoothed_quaternions[i])

    # 转换回旋转矩阵
    smoothed_rotations = []
    for quat in smoothed_quaternions:
        r = R.from_quat(quat)
        smoothed_rotations.append(r.as_matrix())

    return np.array(smoothed_rotations)


def smooth_rotations_double_exponential(rotations, alpha=0.3, beta=0.1):
    """使用双指数平滑（Holt方法）"""
    quaternions = []
    for rot_matrix in rotations:
        try:
            r = R.from_matrix(rot_matrix)
            quaternions.append(r.as_quat())
        except Exception as e:
            print(f"警告: 旋转矩阵转换失败: {e}")
            quaternions.append(np.array([0, 0, 0, 1]))

    quaternions = np.array(quaternions)

    # 确保四元数连续性
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i - 1], quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]

    # 双指数平滑
    smoothed_quaternions = []
    trend = np.zeros(4)
    level = quaternions[0]

    smoothed_quaternions.append(level)

    for i in range(1, len(quaternions)):
        prev_level = level
        level = alpha * quaternions[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

        # 归一化
        smoothed_quat = level / np.linalg.norm(level)
        smoothed_quaternions.append(smoothed_quat)

    # 转换回旋转矩阵
    smoothed_rotations = []
    for quat in smoothed_quaternions:
        r = R.from_quat(quat)
        smoothed_rotations.append(r.as_matrix())

    return np.array(smoothed_rotations)


def smooth_translations_gaussian(translations, sigma=2.0):
    """使用高斯滤波平滑平移"""
    smoothed_translations = np.zeros_like(translations)
    for i in range(3):
        smoothed_translations[:, i] = gaussian_filter1d(translations[:, i], sigma=sigma)
    return smoothed_translations


def smooth_translations_savgol(translations, window_length=15, polyorder=3):
    """使用Savitzky-Golay滤波平滑平移"""
    if window_length >= len(translations):
        window_length = len(translations) - 1 if len(translations) % 2 == 0 else len(translations) - 2
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        window_length = 3

    smoothed_translations = np.zeros_like(translations)
    for i in range(3):
        smoothed_translations[:, i] = savgol_filter(translations[:, i], window_length, polyorder)
    return smoothed_translations


def smooth_translations_exponential(translations, alpha=0.3):
    """使用指数移动平均平滑平移"""
    smoothed_translations = [translations[0]]

    for i in range(1, len(translations)):
        smoothed_trans = (1 - alpha) * smoothed_translations[-1] + alpha * translations[i]
        smoothed_translations.append(smoothed_trans)

    return np.array(smoothed_translations)


def reconstruct_pose_matrices(rotations, translations):
    """重构位姿矩阵"""
    poses = []
    for rot, trans in zip(rotations, translations):
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        poses.append(pose)

    return np.array(poses)


def save_smoothed_poses(pose_files, smoothed_poses, output_folder):
    """保存平滑后的位姿矩阵"""
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for i, (original_file, smoothed_pose) in enumerate(zip(pose_files, smoothed_poses)):
        filename = os.path.basename(original_file)
        output_path = os.path.join(output_folder, filename)
        np.savetxt(output_path, smoothed_pose, fmt='%.6f')
        print(f"已保存: {output_path}")


def calculate_rotation_differences(rotations):
    """计算旋转差异，用于评估抖动程度"""
    differences = []
    for i in range(1, len(rotations)):
        r1 = R.from_matrix(rotations[i - 1])
        r2 = R.from_matrix(rotations[i])
        relative_rot = r2 * r1.inv()
        angle = relative_rot.magnitude()
        differences.append(np.degrees(angle))
    return np.array(differences)


def calculate_translation_differences(translations):
    """计算平移差异"""
    differences = []
    for i in range(1, len(translations)):
        diff = np.linalg.norm(translations[i] - translations[i - 1])
        differences.append(diff)
    return np.array(differences)


def main():
    parser = argparse.ArgumentParser(description='平滑位姿数据以减少旋转抖动')
    parser.add_argument('--input_folder', default='none')
    parser.add_argument('--output_folder', default='none', help='输出文件夹路径')
    parser.add_argument('--method', choices=['exponential', 'gaussian', 'savgol', 'double_exp'],
                        default='gaussian', help='平滑方法')
    parser.add_argument('--rot_strength', type=float, default=2.0,
                        help='旋转平滑强度 (exponential: alpha, gaussian: sigma, savgol: window_size_factor)')
    parser.add_argument('--trans_strength', type=float, default=1.0,
                        help='平移平滑强度')

    args = parser.parse_args()

    # 加载位姿矩阵
    print("正在加载位姿矩阵...")
    pose_files, poses = load_pose_matrices(args.input_folder)

    if len(poses) == 0:
        print("错误: 没有找到有效的位姿文件")
        return

    print(f"成功加载 {len(poses)} 个位姿矩阵")

    # 提取旋转和平移
    rotations, translations = extract_rotation_translation(poses)

    # 计算原始差异
    original_rot_diffs = calculate_rotation_differences(rotations)
    original_trans_diffs = calculate_translation_differences(translations)

    print(
        f"原始旋转抖动 - 平均: {np.mean(original_rot_diffs):.2f}°, 最大: {np.max(original_rot_diffs):.2f}°, 标准差: {np.std(original_rot_diffs):.2f}°")
    print(
        f"原始平移抖动 - 平均: {np.mean(original_trans_diffs):.4f}, 最大: {np.max(original_trans_diffs):.4f}, 标准差: {np.std(original_trans_diffs):.4f}")

    # 根据方法选择平滑算法
    print(f"Smooth with method {args.method}...")

    if args.method == 'exponential':
        smoothed_rotations = smooth_rotations_exponential(rotations, args.rot_strength)
        smoothed_translations = smooth_translations_exponential(translations, args.trans_strength)
    elif args.method == 'gaussian':
        smoothed_rotations = smooth_rotations_gaussian(rotations, args.rot_strength)
        smoothed_translations = smooth_translations_gaussian(translations, args.trans_strength)
    elif args.method == 'savgol':
        window_length = max(3, int(len(poses) * args.rot_strength))
        if window_length % 2 == 0:
            window_length += 1
        smoothed_rotations = smooth_rotations_savgol(rotations, window_length, 3)
        smoothed_translations = smooth_translations_savgol(translations, window_length, 3)
    elif args.method == 'double_exp':
        smoothed_rotations = smooth_rotations_double_exponential(rotations, args.rot_strength, args.trans_strength)
        smoothed_translations = smooth_translations_exponential(translations, args.trans_strength)

    # 重构位姿矩阵
    smoothed_poses = reconstruct_pose_matrices(smoothed_rotations, smoothed_translations)

    # 计算平滑后的差异
    smoothed_rot_diffs = calculate_rotation_differences(smoothed_rotations)
    smoothed_trans_diffs = calculate_translation_differences(smoothed_translations)

    print(
        f"平滑后旋转抖动 - 平均: {np.mean(smoothed_rot_diffs):.2f}°, 最大: {np.max(smoothed_rot_diffs):.2f}°, 标准差: {np.std(smoothed_rot_diffs):.2f}°")
    print(
        f"平滑后平移抖动 - 平均: {np.mean(smoothed_trans_diffs):.4f}, 最大: {np.max(smoothed_trans_diffs):.4f}, 标准差: {np.std(smoothed_trans_diffs):.4f}")

    # 保存结果
    print("Saving smoothed poses...")
    save_smoothed_poses(pose_files, smoothed_poses, args.output_folder)

    print("Complete!")
    rot_improvement = ((np.std(original_rot_diffs) - np.std(smoothed_rot_diffs)) / np.std(original_rot_diffs) * 100)
    trans_improvement = (
                (np.std(original_trans_diffs) - np.std(smoothed_trans_diffs)) / np.std(original_trans_diffs) * 100)
    print(f"Rotation jitter reduce: {rot_improvement:.1f}%")
    print(f"Translation jitter reduce: {trans_improvement:.1f}%")


if __name__ == "__main__":
    main()