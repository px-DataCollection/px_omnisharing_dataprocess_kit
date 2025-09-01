"""visualization_data_sender.py

Read and send Phase 2 data to the simulator.
The program is part of the PX OmniSharing Toolkit.

"""

import socket
import os
import time
import numpy as np
import threading
import json
import argparse
import ast
import h5py

PROJECT_OUTSIDE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_PATH = os.path.join(PROJECT_OUTSIDE_ROOT, "data")

class VisualizeDataSender:
    def __init__(
        self,
        target_clients,
        load_data_path,
        tactile_part_names,
        joint_names,
        update_rate=10,
        data_format="json",
        load_offline_data_flag=True,
        use_left_hand=False,
    ):
        """初始化触觉数据发送器

        Args:
            target_clients: 动作接收端和触觉接收端的客户端列表[{"ip":xxx,"port":xxx},{"ip":xxx,"port":xxx}]
            load_data_path: 离线数据的加载路径
            update_rate: 每秒更新频率
            data_format: 数据格式，'json'
            load_offline_data:是否加载离线数据
            use_left_hand:是否发送左手数据
        """
        self.target_clients = target_clients
        self.update_rate = update_rate
        self.running = False
        self.thread = None
        self.data_format = data_format
        self.loop_offline = False
        self.use_left_hand = use_left_hand
        print(f"target_clients:{target_clients}")

        # 手指触觉模块列表
        self.tactile_part_names = tactile_part_names

        # robot关节列表--已调整为isaacsim控制顺序
        self.joint_names = joint_names

        self.offline_action_data = None
        self.offline_tactile_data = None
        self.offline_num_frames = None
        self.offline_data_file_path = None
        self.loop_offline = None
        if load_offline_data_flag:
            self.offline_data_file_path = load_data_path
            self.loop_offline = True
            self.offline_index = 0
            (
                self.offline_action_data,
                self.offline_tactile_data,
                self.offline_num_frames,
            ) = self.load_offline_data()

        # self.sock = None

        # 网络连接
        self.socks = []

    def get_real_data(self, use_left_hand):  # used for real DH13 hand
        raise NotImplementedError("Error: not implemented")

    def order_tactile_data(self, part_order_raw, tactile_data, handness):
        raise NotImplementedError("Error: not implemented")

    def get_best_fz_x_y(self, part_tactile_data):
        third_col = part_tactile_data[:, 2]  # 合力Fz
        remaining = part_tactile_data[:, 3:]  # 6个按照按压位置及法向力,(num_frame,24)
        quads = []
        for row in remaining:
            row_quads = [row[i : i + 4] for i in range(0, len(row) - 3, 4)]  # (6,4)
            quads.append(row_quads)

        quads = np.array(quads)  # (num_frames,6,4)
        max_quad_idx = np.argmax(
            quads[:, :, 3], axis=1
        )  # 取6个按压位置中法向力最大的x,y
        selected = quads[np.arange(len(part_tactile_data)), max_quad_idx, :2]
        return np.column_stack([third_col, selected])

    def order_joint_data(self, joint_order_raw, joints_data_raw, handness):
        joints_data = [
            joints_data_raw[
                :, np.where(joint_order_raw == f"{handness}_{joint_name}")[0][0]
            ]
            for joint_name in self.joint_names
        ]
        return np.stack(joints_data, axis=1)  # (num_frames,num_joints)

    def load_offline_data(self):
        """
        @brief:加载离线数据，并调整触觉数据和关节数据的拼接顺序
        @return:
            action_data:dict,{"robot_action_data":{"right"：{"joints":...,"pose":...},"left":...}, ...}
            tactile_data:dict,{"tactile_data":{"right":{"$[part_name]":...},"left":...}}
            num_frames:int
        """

        with h5py.File(self.offline_data_file_path, "r") as f:
            obj_pose = None
            lh_pose = None
            lh_joints = None
            rh_pose = []
            rh_joints = []
            tactile_data = {}

            if self.use_left_hand:
                action_data = {
                    "robot_action_data": {"right": {}, "left": {}},
                    "obj_pose": {},
                }
            else:
                action_data = {"robot_action_data": {"right": {}}, "obj_pose": {}}

            # 1. 读取右手关节数据，并调整顺序到仿真控制顺序
            joint_order_raw = f["/dataset/observation/righthand/joints/data"].attrs.get(
                "joint_names", self.joint_names
            )
            rh_joints_raw = f["/dataset/observation/righthand/joints/data"][:]
            num_frames = rh_joints_raw.shape[0]
            rh_joints = self.order_joint_data(joint_order_raw, rh_joints_raw, "right")

            # 2. 读取右手触觉数据，调整触觉数据顺序
            rh_tactile = f["/dataset/observation/righthand/tactile/data"][
                :
            ]  # (num_frames,27*8) (27-->joint_F,press_pos,press_F)
            part_order_raw = f["dataset/observation/righthand/tactile/data"].attrs.get(
                "sensor_names", []
            )
            tactile_data["right"] = self.order_tactile_data(
                part_order_raw, rh_tactile, "right"
            )

            if self.use_left_hand:
                # 1. 读取左手关节数据，并调整顺序到仿真控制顺序
                joint_order_raw = f[
                    "/dataset/observation/lefthand/joints/data"
                ].attrs.get("joint_names", self.joint_names)
                lh_joints_raw = f["/dataset/observation/lefthand/joints/data"][:]
                lh_joints = self.order_joint_data(
                    joint_order_raw, lh_joints_raw, "left"
                )

                # 2. 读取左手触觉数据，调整触觉数据顺序
                lh_tactile = f["/dataset/observation/lefthand/tactile/data"][:]
                part_order_raw = f[
                    "dataset/observation/lefthand/tactile/data"
                ].attrs.get("sensor_names", self.tactile_part_names)
                tactile_data["left"] = self.order_tactile_data(
                    part_order_raw, lh_tactile, "left"
                )
            try:
                obj_pose = f["/dataset/observation/obj1/data"][
                    :
                ]  # (num_frames,7)-->(xyz,qw,qx,qy,qz)
            except:
                print(f"{self.offline_data_file_path} without any obj")

            try:
                rh_pose = f["/dataset/observation/righthand/handpose/data"][:]
            except:
                print(f"{self.offline_data_file_path} without right handpose")
            if self.use_left_hand:
                try:
                    lh_pose = f["/dataset/observation/lefthand/handpose/data"][:]
                except:
                    print(f"{self.offline_data_file_path} without left handpose")

            # 3. 将关节、手部位姿、物体位姿都载入到action_data中
            assert (
                rh_joints.shape[0] == rh_pose.shape[0]
            ), f"右手关节角帧数与右手位姿帧数不相等,\
                {rh_joints.shape[0]}!={rh_pose.shape[0]}"
            if obj_pose is not None:
                assert (
                    rh_joints.shape[0] == obj_pose.shape[0]
                ), f"右手关节角帧数与物体位姿帧数不相等,\
                {rh_joints.shape[0]}!={obj_pose.shape[0]}"
            if lh_pose is not None:
                assert (
                    lh_joints.shape[0] == lh_pose.shape[0]
                ), f"左手关节角帧数与左手位姿帧数不相等,\
                {lh_joints.shape[0]}!={lh_pose.shape[0]}"
                assert (
                    lh_joints.shape[0] == rh_joints.shape[0]
                ), f"左手关节角帧数与右手关节角帧数不相等,\
                {lh_joints.shape[0]}!={rh_joints.shape[0]}"

            action_data["robot_action_data"]["right"]["joints"] = rh_joints
            action_data["robot_action_data"]["right"]["pose"] = rh_pose

            if self.use_left_hand:
                action_data["robot_action_data"]["left"]["joints"] = lh_joints
                action_data["robot_action_data"]["left"]["pose"] = lh_pose

            action_data["obj_pose"] = obj_pose
        # print(f"action data:{action_data}")
        return action_data, tactile_data, num_frames

    def get_tactile_frame_data(self, tactile_frame_data):
        parts_tactile_data_dict = {}
        for i, part in enumerate(self.tactile_part_names):
            parts_tactile_data_dict[part] = (
                tactile_frame_data[i * 3 : i * 3 + 3]
            ).tolist()
        return parts_tactile_data_dict

    def get_next_frame(self, use_left_hand):
        print(
            f"{50*'='} Sending offline data of frame {self.offline_index} {50*'='}"
        )
        """获取下一帧数据"""
        if use_left_hand:
            action_data = {
                "robot_action_data": {"right": {}, "left": {}},
                "obj_pose": {},
            }
            tactile_data = {"tactile_data": {"right": {}, "left": {}}}
        else:
            action_data = {"robot_action_data": {"right": {}}, "obj_pose": {}}
            tactile_data = {"tactile_data": {"right": {}, "left": {}}}
        if self.loop_offline:
            rh_joints = self.offline_action_data["robot_action_data"]["right"][
                "joints"
            ][self.offline_index]
            rh_pose = self.offline_action_data["robot_action_data"]["right"]["pose"][
                self.offline_index
            ]
            # obj_pose = self.offline_action_data["obj_pose"][self.offline_index]
            action_data["robot_action_data"]["right"]["joints"] = rh_joints.tolist()
            action_data["robot_action_data"]["right"]["pose"] = rh_pose.tolist()
            action_data["robot_action_data"]["frame_idx"] = self.offline_index
            # action_data["obj_pose"] = obj_pose.tolist()  # (x,y,z,qw,qx,qy,qz)

            frame_rh_tactile_data = self.offline_tactile_data["right"][
                self.offline_index, :
            ]
            tactile_data["tactile_data"]["right"] = self.get_tactile_frame_data(
                frame_rh_tactile_data
            )

            if self.use_left_hand:
                lh_joints = self.offline_action_data["robot_action_data"]["left"][
                    "joints"
                ][self.offline_index]
                # print(f"offline_action_data:{self.offline_action_data}")
                lh_pose = None
                if (
                    self.offline_action_data["robot_action_data"]["left"]["pose"]
                    is not None
                ):
                    lh_pose = self.offline_action_data["robot_action_data"]["left"][
                        "pose"
                    ][self.offline_index]
                    action_data["robot_action_data"]["left"]["pose"] = lh_pose.tolist()
                else:
                    action_data["robot_action_data"]["left"]["pose"] = []
                    print("warning: there is no left hand pose in hdf5 file")

                action_data["robot_action_data"]["left"]["joints"] = lh_joints.tolist()
                action_data["robot_action_data"]["frame_idx"] = self.offline_index

                frame_lh_tactile_data = self.offline_tactile_data["left"][
                    self.offline_index, :
                ]
                tactile_data["tactile_data"]["left"] = self.get_tactile_frame_data(
                    frame_lh_tactile_data
                )

            self.offline_index += 1
            if self.offline_index >= self.offline_num_frames:
                self.offline_index = 0
                print("LOOP ENDS: return to the first frame")
            return action_data, tactile_data
        else:
            return self.get_real_data(use_left_hand)

    def _update_loop(self):
        """更新循环，定期更新力点并发送数据"""
        start_time = time.time()
        update_interval = 1.0 / self.update_rate

        while self.running:
            current_time = time.time()
            elapsed = current_time - start_time

            action_data, tactile_data = self.get_next_frame(
                self.use_left_hand
            )  # 获取最新的显示数据
            # print(f"发送触觉数据:{tactile_data}")

            if self.data_format in ["json"] and self.socks:
                for i, client in enumerate(self.target_clients):
                    try:
                        if i == 0:
                            json_str = json.dumps(action_data)
                        else:
                            json_str = json.dumps(tactile_data)
                    except Exception as e:
                        print(
                            f"无法将{'动作'if i==0 else '触觉'}数据转化为向json对象:{e}"
                        )

                    target_address_receiver = (client["ip"], client["port"])
                    try:
                        self.socks[i].sendto(
                            json_str.encode("utf-8"), target_address_receiver
                        )
                        # print(f"数据读取和发送花费了{(time.time()-current_time)*1000:.3f}ms")
                    except Exception as e:
                        print(
                            f"failed to send {'action'if i==0 else 'tactile'} data to {client['ip']}:{client['port']}: {e}"
                        )
                    print(
                        f"sending {'action'if i==0 else 'tactile'} data to {client['ip']}:{client['port']}"
                    )

            # 等待到下一个更新周期
            if int(elapsed) % 5 == 0:
                print(
                    f"In robot data: {len(self.joint_names)} joints, and {len(self.tactile_part_names)} sensors are activated. Timestamp: {time.time()}(seconds)"
                )
            time.sleep(max(0, update_interval - (time.time() - current_time)))

    def start(self):
        """根据数据格式启动发送器"""
        if self.running:
            print("robot sender activated")
            return

        if self.data_format not in ["json"]:
            print(f"无效的数据格式: {self.data_format}, 默认使用'json'")
            self.data_format = "json"

        # 根据数据格式创建socket
        try:
            for client in self.target_clients:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socks.append(sock)
            self.running = True
            self.thread = threading.Thread(target=self._update_loop)
            self.thread.daemon = True
            self.thread.start()
            # print("✓ robot触觉数据发送器启动成功")
        except Exception as e:
            print(f"初始化客户端 {client['ip']}:{client['port']} 的socket失败: {e}")
            self.stop()

    def stop(self):
        """安全停止发送器"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        # 安全关闭socket
        if self.socks:
            try:
                for sock in self.socks:
                    sock.close()
            except:
                print(f"关键套接字{sock}产生错误")
                pass
        self.socks = []
        # print("✓ robot触觉数据发送器已停止")

    def get_status(self):
        """获取发送器状态"""
        return {
            "running": self.running,
            "target_client_1": f"{self.target_clients[0]['ip']}:{self.target_clients[0]['port']}",
            "target_client_2": f"{self.target_clients[1]['ip']}:{self.target_clients[1]['port']}",
            "format": self.data_format,
            "rate": self.update_rate,
            "sensors": len(self.tactile_part_names),
            "joints": len(self.joint_names),
        }

class VisualizeDataDexH13Sender(VisualizeDataSender):
    def __init__(
        self,
        target_clients,
        load_data_path,
        update_rate=10,
        data_format="json",
        load_offline_data_flag=True,
        use_left_hand=False,
    ):
        # tactile modules
        self.part_name_to_new_part_name = {
            "thumb_sensor_1": "thumb_tactile_link_0",
            "thumb_sensor_2": "thumb_tactile_link_1",
            "index_sensor_1": "index_tactile_link_0",
            "index_sensor_2": "index_tactile_link_1",
            "index_sensor_3": "index_tactile_link_2",
            "middle_sensor_1": "middle_tactile_link_0",
            "middle_sensor_2": "middle_tactile_link_1",
            "middle_sensor_3": "middle_tactile_link_2",
            "ring_sensor_1": "ring_tactile_link_0",
            "ring_sensor_2": "ring_tactile_link_1",
            "ring_sensor_3": "ring_tactile_link_2",
        }
        self.tactile_part_names = list(self.part_name_to_new_part_name.keys())

        # 16 joints for control
        self.joint_names = [
            "index_joint_0",
            "middle_joint_0",
            "ring_joint_0",
            "thumb_joint_0",
            "index_joint_1",
            "middle_joint_1",
            "ring_joint_1",
            "thumb_joint_1",
            "index_joint_2",
            "middle_joint_2",
            "ring_joint_2",
            "thumb_joint_2",
            "index_joint_3",
            "middle_joint_3",
            "ring_joint_3",
            "thumb_joint_3",
        ]

        super().__init__(
            target_clients=target_clients,
            load_data_path=load_data_path,
            update_rate=update_rate,
            data_format=data_format,
            load_offline_data_flag=load_offline_data_flag,
            use_left_hand=use_left_hand,
            tactile_part_names=self.tactile_part_names,
            joint_names=self.joint_names,
        )

    def order_tactile_data(self, part_order_raw, tactile_data, handness):
        
        try:
            part_names = list(self.part_name_to_new_part_name.keys()) 
            new_part_names = [
                f"{handness}_{v}" for v in list(self.part_name_to_new_part_name.values())
            ]  # eg:right_thumb_tactile_link_0

            # adjust sequence of tactile data
            assert len(part_order_raw) > 0, f"Error: the sensor name is empty"
            access_part_name = part_names
            if "tactile_link" in part_order_raw[0]:
                access_part_name = new_part_names
            tactile_ordered_data = []

            for part in access_part_name:
                part_idx = int(np.asarray(part_idx).item())
                part_tactile_data = tactile_data[:, part_idx * 27 : part_idx * 27 + 27]
                Fz_x_y = self.get_best_fz_x_y(part_tactile_data)
                # print(f"Fz_x,y shape:{Fz_x_y.shape}")
                tactile_ordered_data.append(Fz_x_y)
            tactile_ordered_data = np.concatenate(tactile_ordered_data, axis=1)
        except Exception as e:
            return np.zeros((tactile_data.shape[0], 33))
        
        return tactile_ordered_data

    def get_real_data(self, use_left_hand):  # 发送端接入实际硬件时使用
        pass

# usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sender of action and tactile data from Phase 2")
    parser.add_argument(
        "--clients",
        type=str,
        default='[{"ip": "127.0.0.1", "port": 5679}, {"ip": "127.0.0.1", "port": 5680}]',
        help='List of clients: clients for action and tactile visualization\
            in JSON format. e.g., \'[{"ip":"127.0.0.1","port":5679},{"ip":"127.0.0.1","port":5680}]\'',
    )
    parser.add_argument(
        "--load_offline_data_flag",
        type=bool,
        default=True,
        help="use offline data (instead of real-time robot data)",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default="none",
        help="input data path",
    )
    
    # modify this argument if the current episode does not have left hand
    parser.add_argument(
        "--use_left_hand", type=bool, default=True, help="whether to use the left hand"
    )
    args_cfg = parser.parse_args()

    print("PX Visual -- Phase 2 Data Sender")
    print("=" * 50)
    try:
        clients = json.loads(args_cfg.clients)
    except Exception:
        clients = ast.literal_eval(args_cfg.clients)

    # create the robot data sender
    sender_classes = {
        "dh13": VisualizeDataDexH13Sender,
    }
    sender = None
    for key, cls in sender_classes.items():
        if key in args_cfg.load_data_path:
            sender = cls(
                target_clients=clients,
                update_rate=10,
                data_format="json",
                load_offline_data_flag=args_cfg.load_offline_data_flag,
                load_data_path=args_cfg.load_data_path,
                use_left_hand=args_cfg.use_left_hand,
            )
            break
    if sender is None:
        raise ValueError(f"cannot recognize data type at: {args_cfg.load_data_path}")
    try:
        sender.start()
        print(f"status of sender: {sender.get_status()}")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nABORT signal received")
    except Exception as e:
        print(f"runtime error: {e}")
    finally:
        sender.stop()
        print("Phase 2 Data Sender Terminated")
