"""DexH13_left_new.py

Configuration for the dexterous left hand without sensor

"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from px_replay.config.paths import ROBOT_MODELS_PATH

"""This hand's urdf and usd file, the naming of the joints are different from the tora_one whole body model"""

# Configuration
##
"""DexH13 left hand isaacSim配置"""
DexH13_left_new_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_MODELS_PATH["DEXH13_LEFT_NEW"]["USD"],
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="position"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1),  # 调整初始位置
        rot=(0.0, 0.0, 0.7071, 0.7071),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            # 拇指关节
            "left_thumb_joint_0": 0.00,  # 拇指侧摆
            "left_thumb_joint_1": 0.00,  # 拇指近端旋转
            "left_thumb_joint_2": 0.00,  # 拇指远端旋转
            "left_thumb_joint_3": 0.00,  # 拇指掌端旋转
            # 食指关节
            "left_index_joint_0": 0.0,  # 食指侧摆
            "left_index_joint_1": 0.0,  # 食指近端旋转
            "left_index_joint_2": 0.0,  # 食指远端旋转
            "left_index_joint_3": 0.0,  # 食指掌端旋转
            # 中指关节
            "left_middle_joint_0": 0.0,  # 中指侧摆
            "left_middle_joint_1": 0.0,  # 中指近端旋转
            "left_middle_joint_2": 0.0,  # 中指远端旋转
            "left_middle_joint_3": 0.0,  # 中指掌端旋转
            # 无名指关节
            "left_ring_joint_0": 0.0,  # 无名指侧摆
            "left_ring_joint_1": 0.0,  # 无名指近端旋转
            "left_ring_joint_2": 0.0,  # 无名指远端旋转
            "left_ring_joint_3": 0.0,  # 无名指掌端旋转
        },
        joint_vel={".*": 0.0},  # 所有关节初始速度为0
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
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
            effort_limit=20,
            velocity_limit=50.0,
            stiffness={
                "left_thumb_joint_0": 1000,
                "left_thumb_joint_1": 1000,
                "left_thumb_joint_2": 1000,
                "left_thumb_joint_3": 1000,
                "left_index_joint_0": 1000,
                "left_index_joint_1": 1000,
                "left_index_joint_2": 1000,
                "left_index_joint_3": 1000,
                "left_middle_joint_0": 1000,
                "left_middle_joint_1": 1000,
                "left_middle_joint_2": 1000,
                "left_middle_joint_3": 1000,
                "left_ring_joint_0": 1000,
                "left_ring_joint_1": 1000,
                "left_ring_joint_2": 1000,
                "left_ring_joint_3": 1000,
            },
            damping={
                "left_thumb_joint_0": 1000,
                "left_thumb_joint_1": 1000,
                "left_thumb_joint_2": 1000,
                "left_thumb_joint_3": 1000,
                "left_index_joint_0": 1000,
                "left_index_joint_1": 1000,
                "left_index_joint_2": 1000,
                "left_index_joint_3": 1000,
                "left_middle_joint_0": 1000,
                "left_middle_joint_1": 1000,
                "left_middle_joint_2": 1000,
                "left_middle_joint_3": 1000,
                "left_ring_joint_0": 1000,
                "left_ring_joint_1": 1000,
                "left_ring_joint_2": 1000,
                "left_ring_joint_3": 1000,
            },
            # armature={
            #     "mzcb": 0.001,
            #     "mzjdxz": 0.001,
            #     "mzzdxz": 0.001,
            #     "mzyd": 0.001,
            #     "szcb": 0.001,
            #     "szjdxz": 0.001,
            #     "szzdxz": 0.001,
            #     "szydxz": 0.001,
            #     "zzcb": 0.001,
            #     "zzjdxz": 0.001,
            #     "zzzdxz": 0.001,
            #     "zzydxz": 0.001,
            #     "wmzcb": 0.001,
            #     "wmzjdxz": 0.001,
            #     "wmzzdxz": 0.001,
            #     "wmzydxz": 0.001,
            # },
        ),
    },
)
