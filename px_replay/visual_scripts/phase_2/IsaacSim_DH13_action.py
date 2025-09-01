"""IsaacSim_DexH13_action.py

DexH13 Action Visualization
Replay the Phase 2 data in simulation

Start the data sender (visualization_data_sender) to generate mock data
or plug in physical data before running this script.

This program is part of the PX OmniSharing Toolkit.

"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="DexH13动作可视化演示程序")
parser.add_argument("--num_envs", type=int, default=1, help="生成的环境数量")
parser.add_argument(
    "--object_type",
    type=str,
    default="none",
    help="deprecated", 
)
parser.add_argument(
    "--use_left_hand", type=bool, default=True, help="是否显示左手动作"
)
parser.add_argument(
    "--host_ip",
    type=str,
    default="127.0.0.1",
    help="如果显示动作和触觉的电脑不一样，需要填入电脑实际ip",
)
# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# DexH13设备类型（硬编码）
DEVICE_TYPE = "DEXH13"

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sharklog

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim import (
    SimulationContext,
    SimulationCfg,
    GroundPlaneCfg,
    DomeLightCfg,
    PinholeCameraCfg,
)

from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import CameraCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_matrix

# 预定义配置
from px_replay.robot.DexH13.DexH13_right_new import (
    DexH13_right_new_CFG,
)
from px_replay.robot.DexH13.DexH13_left_new import (
    DexH13_left_new_CFG,
)

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.math import matrix_from_quat
import os
import math

# tactile data process
from px_replay.dataprocess.tactile.device_interfaces import DeviceType
from px_replay.dataprocess.tactile.device_manager import DeviceManager

from px_replay.dataprocess.tactile.tactile_ForceFieldGenerator import (
    ForceFieldGenerator,
)

import json

class DexH13ActionVisualization:
    """
    DexH13动作可视化系统
    该系统负责：
    1. 控制16个关节的运动
    """

    def __init__(self, sim_device: str, host_ip: str = "127.0.0.1"):
        """
        初始化

        Args:
            sim_device (str): 仿真设备 ("cuda" 或 "cpu")
        """
        self.device_type = DeviceType.DEXH13
        self.sim_device = sim_device

        self.device_manager = DeviceManager(self.device_type)

        component_kwargs = {
            "receiver_kwargs": {
                "ip": host_ip,
                "port": 5679,
            },
            "grid_kwargs": {"device": self.sim_device},
            "joint_kwargs": {},
            "driver_kwargs": {},
        }

        try:
            # 初始化所有组件
            self.device_manager.initialize_all_components(**component_kwargs)
        except TypeError as e:
            sharklog.warning(f"参数错误，尝试使用默认参数: {e}")
            # 如果参数有问题，尝试不传递任何参数
            self.device_manager.initialize_all_components()

        # 获取各组件的引用
        self.data_receiver = self.device_manager.get_data_receiver()  # 数据接收器

        # joint_name在scene中对应的joint_id的映射
        self.joint_name_joint_to_ids = {}

        # # DexH13 joint_name在scene中joint_id的映射 left_hand
        self.joint_name_joint_to_ids["left"] = {
            "left_thumb_joint_0": 3,
            "left_thumb_joint_1": 7,
            "left_thumb_joint_2": 11,
            "left_thumb_joint_3": 15,
            "left_index_joint_0": 0,
            "left_index_joint_1": 4,
            "left_index_joint_2": 8,
            "left_index_joint_3": 12,
            "left_middle_joint_0": 1,
            "left_middle_joint_1": 5,
            "left_middle_joint_2": 9,
            "left_middle_joint_3": 13,
            "left_ring_joint_0": 2,
            "left_ring_joint_1": 6,
            "left_ring_joint_2": 10,
            "left_ring_joint_3": 14,
        }

        self.num_joints = len(self.joint_name_joint_to_ids["left"])
        # DexH13 joint_name到body_id的映射 right_hand
        # TODO:debug左手是否一样的映射
        self.joint_name_joint_to_ids["right"] = {
            "right_thumb_joint_0": 3,
            "right_thumb_joint_1": 7,
            "right_thumb_joint_2": 11,
            "right_thumb_joint_3": 15,
            "right_index_joint_0": 0,
            "right_index_joint_1": 4,
            "right_index_joint_2": 8,
            "right_index_joint_3": 12,
            "right_middle_joint_0": 1,
            "right_middle_joint_1": 5,
            "right_middle_joint_2": 9,
            "right_middle_joint_3": 13,
            "right_ring_joint_0": 2,
            "right_ring_joint_1": 6,
            "right_ring_joint_2": 10,
            "right_ring_joint_3": 14,
        }
        sharklog.info(f"DexH13 model initializes, supporting {self.num_joints} joints for control")

    def initialize_visualization(self, use_left_hand):
        """
        初始化所有部件的可视化标记

        Args:
            sim: 仿真上下文
        """
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.obj_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_obj")
        )
        self.rh_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_rh")
        )
        if use_left_hand:
            self.lh_marker = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/ee_lh")
            )
        sharklog.info("All visualization markers initialize")

    def update_action_controls(self, work_obj, left_hand, right_hand, sim_time):
        # 获取最新接收的数据
        latest_data = self.data_receiver.get_latest_data()
        if latest_data is not None:
            # 提取物体位姿数据
            obj_pose = None
            if work_obj is not None and "obj_pose" in latest_data:
                obj_pose = latest_data["obj_pose"]
            if obj_pose is not None:
                self.update_obj_action_control(work_obj, obj_pose)

            # 提取机器人的位姿及关节数据
            action_data = None
            if "robot_action_data" in latest_data:
                action_data = latest_data["robot_action_data"]
            if action_data is not None:
                if left_hand is not None:
                    if "left" in action_data:
                        self.update_robot_action_control(
                            left_hand, "left", action_data, sim_time
                        )
                    else:
                        sharklog.warning(f"Cannot receive pose and joint data fron LH")
                if "right" in action_data:
                    self.update_robot_action_control(
                        right_hand, "right", action_data, sim_time
                    )
                else:
                    sharklog.warning(f"Cannot receive pose and joint data fron RH")
        else:
            sharklog.warning("No pose or joint data from the robot")

    def get_7d_pose(self, pose):
        """
        (x,y,z,qx,qy,qz,qw)->(x,y,z,qw,qx,qy,qz)
        """
        if isinstance(pose, np.array) and pose.shape == (7,):
            pose_data_7d = np.zeros(7)
            pose_data_7d[0:3] = pose[0:3]
            pose_data_7d[3] = pose[-1]
            pose_data_7d[4:-1] = pose[3:-1]
            return pose_data_7d

    def update_obj_action_control(self, obj, obj_pose):
        """
        基于接收到的数据更新物体的位姿控制

        Args:
            obj: 物体对象
            sim_time: 仿真时间
        """
        if obj_pose is not None:
            obj_pose_raw = np.array(obj_pose)
            # obj_pose_7d = self.get_7d_pose(obj_pose_raw)
            obj_pose_7d = obj_pose_raw
            obj_pose_tensor = torch.tensor(
                np.array([obj_pose_7d]),
                dtype=torch.float32,
                device=self.sim_device,
            )
            obj.write_root_pose_to_sim(root_pose=obj_pose_tensor)
            self.obj_marker.visualize(obj_pose_tensor[:, 0:3], obj_pose_tensor[:, 3:7])
        else:
            sharklog.warning("没有收到物体位姿数据")

    def update_robot_action_control(self, robot, handness, latest_data, sim_time):
        """
        基于接收到的数据更新机器人关节控制和位姿控制

        Args:
            robot: 机器人对象
            sim_time: 仿真时间
        """
        if latest_data is not None and handness in latest_data:
            # sharklog.debug(
            #     f"{handness} hand receives motion data: {latest_data[handness]}"
            # )

            try:
                # 提取关节数据和位姿数据
                joint_data = latest_data[handness]["joints"]  # list
                pose_data_raw = np.array(latest_data[handness]["pose"])  # list
                # pose_data_7d = self.get_7d_pose(pose_data_raw)
                pose_data_7d = np.array(pose_data_raw)
                hand_pose_tensor = torch.tensor(
                    np.array([pose_data_7d]),
                    dtype=torch.float32,
                    device=self.sim_device,
                )

                # 更新位姿和关节数据
                robot.write_root_pose_to_sim(root_pose=hand_pose_tensor)
                robot.set_joint_position_target(
                    torch.tensor(joint_data, device=self.sim_device)
                )
                if handness == "left":
                    self.lh_marker.visualize(
                        hand_pose_tensor[:, 0:3], hand_pose_tensor[:, 3:7]
                    )
                elif handness == "right":
                    self.rh_marker.visualize(
                        hand_pose_tensor[:, 0:3], hand_pose_tensor[:, 3:7]
                    )
            except Exception as e:
                sharklog.warning(f"关节控制处理失败: {e}")
        else:
            pass

    def start_data_reception(self):
        """开始从设备接收数据"""
        try:
            self.data_receiver.start_receiver()
            sharklog.info("✓ DexH13 Data Receiver Activated")
        except Exception as e:
            sharklog.warning(f"启动数据接收器失败，使用模拟数据: {e}")

    def cleanup(self):
        """清理所有资源"""
        try:
            self.device_manager.cleanup()
            sharklog.info("✓ DexH13动作可视化系统清理完成")
        except Exception as e:
            sharklog.error(f"清理过程中出错: {e}")


def run_simulator(
    sim: sim_utils.SimulationContext, scene: InteractiveScene, use_left_hand, host_ip
):
    """
    运行仿真器 - DexH13版本

    Args:
        sim: 仿真上下文
        scene: 交互场景
    """
    sharklog.info("DexH13 Robot server is running...")
    sharklog.info("DexH13 with action vis demo ")




    # 获取机器人对象  left_hand
    right_hand = None
    left_hand = None
    if use_left_hand:
        left_hand = scene["left_hand"]
        sharklog.debug(f"left_hand joints: {left_hand.data.joint_names}")
        sharklog.debug(f"left_hand links: {left_hand.data.body_names}")

    right_hand = scene["right_hand"]
    sharklog.debug(f"right_hand joints: {right_hand.data.joint_names}")
    sharklog.debug(f"right_hand links: {right_hand.data.body_names}")
    work_obj = None

    # 配置机器人实体 - 更新为DexH13的关节和部件名称
    if use_left_hand:
        robot_entity_cfg_left = SceneEntityCfg(
            "left_hand",
            joint_names=[
                # 16个关节名称列表
                "left_thumb_joint_0",
                "left_thumb_joint_1",
                "left_thumb_joint_2",
                "left_thumb_joint_3",
                "left_index_joint_0",
                "left_index_joint_1",
                "left_index_joint_2",
                "left_index_joint_3",
                "left_middle_joint_0",
                "left_middle_joint_1",
                "left_middle_joint_2",
                "left_middle_joint_3",
                "left_ring_joint_0",
                "left_ring_joint_1",
                "left_ring_joint_2",
                "left_ring_joint_3",
            ],
        )
    robot_entity_cfg_right = SceneEntityCfg(
        "right_hand",
        joint_names=[
            # 16个关节名称列表
            "right_thumb_joint_0",
            "right_thumb_joint_1",
            "right_thumb_joint_2",
            "right_thumb_joint_3",
            "right_index_joint_0",
            "right_index_joint_1",
            "right_index_joint_2",
            "right_index_joint_3",
            "right_middle_joint_0",
            "right_middle_joint_1",
            "right_middle_joint_2",
            "right_middle_joint_3",
            "right_ring_joint_0",
            "right_ring_joint_1",
            "right_ring_joint_2",
            "right_ring_joint_3",
        ],
    )

    # 初始化DexH13动作可视化系统
    action_system = DexH13ActionVisualization(sim.device, host_ip)

    # 启动数据接收
    action_system.start_data_reception()  # 接收位姿和关节数据

    # 初始化可视化组件
    action_system.initialize_visualization(use_left_hand)

    # 验证body_ids映射的正确性
    print("++++++++", scene)
    if use_left_hand:
        robot_entity_cfg_left.resolve(scene)
        print(robot_entity_cfg_right)
    robot_entity_cfg_right.resolve(scene)
    print(robot_entity_cfg_right)

    for (
        handness,
        joint_name_joint_to_id,
    ) in action_system.joint_name_joint_to_ids.items():
        for joint_name, joint_id in joint_name_joint_to_id.items():
            if handness == "right":
                assert (
                    joint_id in robot_entity_cfg_right.joint_ids
                ), f"joint {joint_name} from {handness} hand does not have body_id {joint_id}: not in robot_entity_cfg.joint_ids!"
            elif use_left_hand:
                assert (
                    joint_id in robot_entity_cfg_left.joint_ids
                ), f"joint {joint_name} from {handness} hand does not have body_id {joint_id}: not in robot_entity_cfg.joint_ids!"
            sharklog.debug(
                f"joint name mapping from {handness} hand to body_id: {action_system.joint_name_joint_to_ids[handness]}"
            )

    # 仿真环境设置
    sim_dt = sim.get_physics_dt()  # 获取物理仿真时间步长
    print(f"sim_dt:{sim_dt}")

    if left_hand is not None:
        left_root_state = left_hand.data.default_root_state.clone()  # 克隆默认根状态
        left_root_state[:, :3] += scene.env_origins  # 添加环境原点偏移
        left_hand.write_root_state_to_sim(left_root_state)  # 写入根状态到仿真
    if right_hand is not None:
        right_root_state = right_hand.data.default_root_state.clone()
        right_root_state[:, :3] += scene.env_origins
        right_hand.write_root_state_to_sim(right_root_state)

    # 清理内部缓冲区并重置场景
    scene.reset()
    sharklog.info("Scene reset. Simulation initiates...")
    sim_time = 0.0

    try:
        # 主仿真循环
        while simulation_app.is_running():
            # 1. 更新位姿和关节控制（基于接收的数据或生成平滑运动)
            action_system.update_action_controls(
                work_obj, left_hand, right_hand, sim_time
            )

            # 2. 将控制数据写入仿真环境
            scene.write_data_to_sim()

            # 3. 确保GPU同步（避免异步执行问题）
            torch.cuda.synchronize()

            # 4. 执行仿真步进
            sim.step()
            sim_time += sim_dt
            scene.update(sim_time)

    except KeyboardInterrupt:
        sharklog.info("Simulation terminated by the user")
    except Exception as e:
        sharklog.error(f"Error in simulation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理所有资源
        action_system.cleanup()
        sharklog.info("Clean-up complete")


from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from IsaacSim_replay_utils import (
    WORK_BENCH_MODEL_CONFIG,
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    def __init__(
        self,
        lh_cfg,
        rh_cfg,
        object_cfg,
        work_bench_cfg,
        num_envs,
        env_spacing,
        **kwargs,
    ):
        super().__init__(num_envs=num_envs, env_spacing=env_spacing, **kwargs)
        self.dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.5, 0.5, 0.5)),
        )

        camera_pos = [-0.77296, -0.57612, -0.53844]
        # 相机
        self.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=720,
            width=720,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "semantic_segmentation",
                "instance_id_segmentation_fast",
            ],
            spawn=PinholeCameraCfg(
                focal_length=15.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=list(camera_pos),
                convention="world",
            ),
            depth_clipping_behavior="min",
        )

        self.left_hand = None
        self.right_hand = None
        if lh_cfg is not None:
            self.left_hand = lh_cfg
        if rh_cfg is not None:
            self.right_hand = rh_cfg
        if object_cfg is not None:
            self.object = object_cfg
        if work_bench_cfg is not None:
            self.work_bench = work_bench_cfg

def main():
    """
    主函数 - 程序入口点

    功能:
    1. 初始化日志系统
    2. 配置仿真环境
    3. 设置相机视角
    4. 创建交互场景
    5. 启动仿真循环
    """
    # 初始化日志系统（启用调试模式）
    sharklog.init(debug=True)

    # 加载仿真配置
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 设置相机视角（优化观察角度）
    sim.set_camera_view(
        eye=[-0.77296, -0.57612, -0.53844],
        target=[0.5, -0.2, 3.0],  # 相机位置  # 观察目标点
    )

    # 获取采集房间模型配置
    work_bench_config = WORK_BENCH_MODEL_CONFIG["work_bench"]
    sharklog.info(f"Path to the workbench: {work_bench_config['usd_path']}")
    sharklog.info(f"Workbench model config: scale={work_bench_config['scale']}")
    sharklog.info(f"Workbench model config: mass={work_bench_config['mass']}")
    sharklog.info(f"Workbench model config: init_pos={work_bench_config['init_pos']}")
    sharklog.info(
        f"Workbench model config: collision_enabled={work_bench_config.get('collision_enabled')}"
    )
    sharklog.info(
        f"Workbench model config: rigid_body_enabled={work_bench_config.get('rigid_body_enabled')}"
    )
    sharklog.info(
        f"Workbench model config: kinematic_enabled={work_bench_config.get('kinematic_enabled')}"
    )
    collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=work_bench_config.get("collision_enabled", False),
        contact_offset=0.05,  # 设置接触偏移为0.05
        rest_offset=0.0,  # 设置静止偏移为0
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=work_bench_config.get("kinematic_enabled", True),
        rigid_body_enabled=work_bench_config.get("rigid_body_enabled", True),
        disable_gravity=True,  # 禁用重力
        retain_accelerations=False,  # 不保留加速度
        linear_damping=0.0,  # 线性阻尼
        angular_damping=0.0,  # 角阻尼
    )
    work_bench_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/work_bench",
        spawn=sim_utils.UsdFileCfg(
            usd_path=work_bench_config["usd_path"],
            scale=work_bench_config["scale"],
            mass_props=sim_utils.MassPropertiesCfg(mass=work_bench_config["mass"]),
            collision_props=collision_props,
            rigid_props=rigid_props,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.3, -0.4, -0.4),
        ),
    )
    sharklog.info("WORKBENCH CREATED")
    sharklog.info(
        "Collision config: " + str(work_bench_cfg.spawn.collision_props.collision_enabled)
    )
    sharklog.info(
        "Rigid body config: " + str(work_bench_cfg.spawn.rigid_props.kinematic_enabled)
    )
    sharklog.info(
        "Rigit body status: " + str(work_bench_cfg.spawn.rigid_props.rigid_body_enabled)
    )
    sharklog.info(
        "Gravity prohibition status:" + str(work_bench_cfg.spawn.rigid_props.disable_gravity)
    )
    # 加载手部模型
    right_hand_cfg = None
    left_hand_cfg = None
    if args_cli.use_left_hand:
        left_hand_cfg = DexH13_left_new_CFG.replace(
            prim_path="{ENV_REGEX_NS}/left_hand",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0, 0.06, 0),  # 调整初始位置
                rot=(-0.70711, 0, 0.70711, 0),  # 初始旋转
                lin_vel=(0.0, 0.0, 0.0),  # 线速度
                ang_vel=(0.0, 0.0, 0.0),  # 角速度
                joint_pos={
                    # DexH13关节初始位置 - 16个关节
                    "left_thumb_joint_0": 0.0,
                    "left_thumb_joint_1": 0.020,
                    "left_thumb_joint_2": 0.020,
                    "left_thumb_joint_3": 0.020,
                    "left_index_joint_0": 0.0,
                    "left_index_joint_1": 0.020,
                    "left_index_joint_2": 0.020,
                    "left_index_joint_3": 0.020,
                    "left_middle_joint_0": 0.0,
                    "left_middle_joint_1": 0.020,
                    "left_middle_joint_2": 0.020,
                    "left_middle_joint_3": 0.020,
                    "left_ring_joint_0": 0.0,
                    "left_ring_joint_1": 0.020,
                    "left_ring_joint_2": 0.020,
                    "left_ring_joint_3": 0.020,
                },
                joint_vel={".*": 0.0},  # 所有关节初始速度为0
            ),
        )
    right_hand_cfg = DexH13_right_new_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0.06, 0),  # 调整初始位置
            rot=(-0.70711, 0, 0.70711, 0),  # 初始旋转
            lin_vel=(0.0, 0.0, 0.0),  # 线速度
            ang_vel=(0.0, 0.0, 0.0),  # 角速度
            joint_pos={
                # DexH13关节初始位置 - 16个关节
                "right_thumb_joint_0": 0.0,
                "right_thumb_joint_1": 0.020,
                "right_thumb_joint_2": 0.020,
                "right_thumb_joint_3": 0.020,
                "right_index_joint_0": 0.0,
                "right_index_joint_1": 0.020,
                "right_index_joint_2": 0.020,
                "right_index_joint_3": 0.020,
                "right_middle_joint_0": 0.0,
                "right_middle_joint_1": 0.020,
                "right_middle_joint_2": 0.020,
                "right_middle_joint_3": 0.020,
                "right_ring_joint_0": 0.0,
                "right_ring_joint_1": 0.020,
                "right_ring_joint_2": 0.020,
                "right_ring_joint_3": 0.020,
            },
            joint_vel={".*": 0.0},  # 所有关节初始速度为0
        ),
    )
    scene_cfg = RobotSceneCfg(
        lh_cfg=left_hand_cfg,
        rh_cfg=right_hand_cfg,
        object_cfg=None,
        work_bench_cfg=work_bench_cfg,
        num_envs=args_cli.num_envs,
        env_spacing=2.0,  # 环境数量  # 环境间距
    )
    scene = InteractiveScene(scene_cfg)

    # 重置仿真环境
    sim.reset()
    sharklog.info("Simulation Environment Set")

    # 启动仿真器
    run_simulator(sim, scene, args_cli.use_left_hand, args_cli.host_ip)

if __name__ == "__main__":
    """
    Entry point

    Process:
    1. Parse the arguments from the command line
    2. Starts Omniverse
    3. Initialize DexH13 action visualization
    4. Run the main simulation loop
    5. Clean the occupied resource and exit
    """
    main()
    simulation_app.close()
