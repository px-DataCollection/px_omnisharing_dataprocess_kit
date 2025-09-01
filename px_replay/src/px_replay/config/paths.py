import os

PROJECT_ROOT = os.environ.get("PX_REPLAY_PATH")
assert PROJECT_ROOT is not None, "Please set up PX_REPLAY_PATH"

print(f"=== px_replay project root at {PROJECT_ROOT} ===")
PROJECT_OUTSIDE = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# 定义各个模块的路径
ROBOT_PATH = os.path.join(PROJECT_ROOT, "robot")
SIMULATOR_PATH = os.path.join(PROJECT_ROOT, "simulator")
EXAMPLE_PATH = os.path.join(PROJECT_ROOT, "example")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
EXAMPLE_DATA_PATH = os.path.join(DATA_PATH, "example_data")
LEARNING_PATH = os.path.join(PROJECT_ROOT, "learning")
TASK_PATH = os.path.join(PROJECT_ROOT, "task")
STANDALONE_PATH = os.path.join(PROJECT_ROOT, "standalone")
# PX_TACTILE_SENSOR_PATH = os.path.join(PROJECT_ROOT, "robot/px_tactile_sensor")
ASSET_PATH = os.path.join(PROJECT_OUTSIDE, "asset")
DATA_PATH = os.path.join(PROJECT_OUTSIDE, "data")

TACTILE_PATH = os.path.join(PROJECT_ROOT, "dataprocess/tactile")

# 机器人模型文件路径
ROBOT_MODELS_PATH = {
    "TORA": {
        "USD": os.path.join(ROBOT_PATH, "tora", "pr_tora_one_dev.usd"),
        "USD_UNICOLOR": os.path.join(ROBOT_PATH, "tora", "pr_tora_one_dev_color.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "tora", "tora_config.yaml"),
        "URDF_RARM": os.path.join(ROBOT_PATH, "tora", "pr_tora_onerarm_dev.urdf"),
        "URDF_LARM": os.path.join(ROBOT_PATH, "tora", "pr_tora_onelarm_dev.urdf"),
        "URDF": os.path.join(ROBOT_PATH, "tora", "pr_tora_one_dev.urdf"),
        "MESH": os.path.join(ROBOT_PATH, "tora", "meshes"),
    },
    "DEXH13": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_right.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", "DexH13_right_config.yaml"),
    },
    "DEXH13_RIGHT": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_right.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", ". .yaml"),
    },
    "DEXH13_LEFT": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_left.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", "DexH13_left_config.yaml"),
    },
    "DEXH13_RIGHT_WITH_CONTACT_SENSOR": {
        "USD": os.path.join(
            ROBOT_PATH, "DexH13", "DexH13_right_contact_sensors-demo.usd"
        ),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", ". .yaml"),
    },
    "DEXH13_CENTER_SENSOR": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_right_center_sensors.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", ". .yaml"),
    },
    "DATA_GLOVE_RIGHT_V1": {
        # "USD": os.path.join(ROBOT_PATH, "DataGlove", "DataGlove_right.usd"),
        "USD": os.path.join(ROBOT_PATH, "DataGlove", "v1", "DataGlove_right.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DataGlovs", ". .yaml"),
    },
    "DATA_GLOVE_RIGHT_V0": {
        # "USD": os.path.join(ROBOT_PATH, "DataGlove", "DataGlove_right.usd"),
        "USD": os.path.join(ROBOT_PATH, "DataGlove", "v0", "DataGlove_right.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DataGlovs", ". .yaml"),
    },
    "DATA_GLOVE_LEFT_V1": {
        # "USD": os.path.join(ROBOT_PATH, "DataGlove", "DataGlove_right.usd"),
        "USD": os.path.join(ROBOT_PATH, "DataGlove", "v1", "DataGlove_left.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DataGlovs", ". .yaml"),
    },
    "DATA_GLOVE_LEFT_V0": {
        # "USD": os.path.join(ROBOT_PATH, "DataGlove", "DataGlove_right.usd"),
        "USD": os.path.join(ROBOT_PATH, "DataGlove", "v0", "DataGlove_left.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DataGlovs", ". .yaml"),
    },
    "DEXH13_LEFT_WITHOUT_SENSOR": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_left_without_sensor.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", ". .yaml"),
    },
    "DEXH13_RIGHT_WITHOUT_SENSOR": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_right_without_sensor.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", ". .yaml"),
    },
    "DEXH13_LEFT_NEW": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_left_new.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", ". .yaml"),
    },
    "DEXH13_RIGHT_NEW": {
        "USD": os.path.join(ROBOT_PATH, "DexH13", "DexH13_right_new.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH13", ". .yaml"),
    },
    "DEXH5_RIGHT": {
        "USD": os.path.join(ROBOT_PATH, "DexH5", "DexH5_right.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH5", ". .yaml"),
    },
    "DEXH5_LEFT": {
        "USD": os.path.join(ROBOT_PATH, "DexH5", "DexH5_left.usd"),
        "CONFIG": os.path.join(ROBOT_PATH, "DexH5", ". .yaml"),
    },
}


# 重播数据路径
GLOVE_DATA_PATH = os.path.join(DATA_PATH, "data_format_one_dot_two_dot_one")
RETARGET_DATA_PATH = os.path.join(DATA_PATH, "data_format_one_dot_two_dot_two")

### Just for the demo example
DEFAULT_ENV_CONFIG_PATH = os.path.join(CONFIG_PATH, "congfig_two_tora.yaml")
DEFAULT_ENV_CONFIG_SCENE = os.path.join(CONFIG_PATH, "config_scene.yaml")


def get_abs_path(relative_path: str) -> str:
    """将相对于项目根目录的路径转换为绝对路径"""
    return os.path.join(PROJECT_ROOT, relative_path)
