"""
adapters_Dexh13.py

DexH13 Device Adapter Implementation

This module implements the DexH13-specific adapters that conform to the
device interface contracts. It provides concrete implementations for data
reception, grid management, joint control, and device communication for
DexH13 robotic hand hardware.

Key Features:
- UDP-based real-time data reception with threading support
- Advanced grid management with coordinate transformation and rotation
- Joint mapping compatible with Tora robotic hand interface
- Mock driver for testing and simulation purposes

"""

import json
import os
import torch
import socket
import threading

import numpy as np
from typing import Dict, List, Optional, Tuple

from px_replay.dataprocess.tactile.device_interfaces import (
    IDataReceiver,
    IGridManager,
    IJointController,
    IDeviceDriver,
)


class DexH13DataReceiver(IDataReceiver):
    """
    DexH13 Data Receiver Adapter

    Implements threaded UDP data reception for DexH13 device.
    Provides real-time tactile and joint data reception with
    non-blocking operation and automatic data parsing.
    """

    def __init__(self, ip="127.0.0.1", port=5679):
        """
        Initialize DexH13 data receiver

        Args:
            ip (str): UDP server IP address
            port (int): UDP server port number
        """
        self.server_address = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.server_address)
        self.sock.settimeout(0.1)  # Non-blocking with timeout
        self.latest_data = None
        self.running = False
        self.receiver_thread = None

    def start_receiver(self):
        """
        Start threaded data reception

        Creates and starts a daemon thread for continuous data reception
        without blocking the main application thread.
        """
        if not self.running:
            self.running = True
            self.receiver_thread = threading.Thread(target=self._receive_loop)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

    def _receive_loop(self):
        """
        Internal data reception loop

        Continuously receives and parses UDP data in separate thread.
        Handles JSON parsing and error recovery automatically.
        """
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                data_str = data.decode("utf-8")
                self.latest_data = json.loads(data_str)
            except socket.timeout:
                continue  # Continue listening
            except json.JSONDecodeError:
                continue  # Skip malformed data
            except Exception:
                pass  # Handle other exceptions gracefully

    def get_latest_data(self) -> Optional[Dict]:
        """
        Get latest received data

        Returns:
            Optional[Dict]: Latest parsed data or None if no data available
        """
        return self.latest_data

    def stop_receiver(self):
        """Stop data reception and cleanup resources"""
        self.running = False
        if self.sock:
            self.sock.close()


class DexH13GridManager(IGridManager):
    """
    DexH13 Grid Manager Adapter

    Advanced grid management for DexH13 tactile sensors.
    Supports coordinate loading, 3D transformations, and
    device-specific rotations for accurate sensor positioning.
    """

    def __init__(self, device=None):
        """
        Initialize DexH13 grid manager

        Args:
            device (str): Computing device ('cpu' or 'cuda')
        """
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.custom_grid_points = {}
        self.coords_data = self.load_coords()

        # DexH13-specific rotation configurations for different finger parts
        self.x_rotation_parts = {}
        self.y_rotation_parts = {}
        self.z_rotation_parts = {}

    def load_coords(self) -> Dict:
        """
        Load DexH13 coordinate data from configuration file

        Returns:
            Dict: Coordinate data for all DexH13 finger parts
        """
        from px_replay.config.paths import ROBOT_PATH

        grid_coords_file = os.path.join(
            ROBOT_PATH, "DexH13", "tactile_data", "grid_coords.json"
        )

        try:
            with open(grid_coords_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def create_default_grid(self, part_name: str) -> Tuple[torch.Tensor, List]:
        """
        Create default grid points for DexH13 parts with transformations

        Args:
            part_name (str): Name of the finger part (e.g., 'szyd_s', 'mzyd_s')

        Returns:
            Tuple[torch.Tensor, List]: Transformed grid points and original coordinates
        """

        # DexH13-specific part coordinate mapping
        coords_mapping = {
            "thumb_sensor_2": self.coords_data.get("mzyd_s", []),
            "thumb_sensor_1": self.coords_data.get("mzzd_s", []),
            "index_sensor_3": self.coords_data.get("yd_s", []),
            "index_sensor_2": self.coords_data.get("zd_s", []),
            "index_sensor_1": self.coords_data.get("jd_s", []),
            "middle_sensor_3": self.coords_data.get("yd_s", []),
            "middle_sensor_2": self.coords_data.get("zd_s", []),
            "middle_sensor_1": self.coords_data.get("jd_s", []),
            "ring_sensor_3": self.coords_data.get("yd_s", []),
            "ring_sensor_2": self.coords_data.get("zd_s", []),
            "ring_sensor_1": self.coords_data.get("jd_s", []),
        }

        if part_name not in coords_mapping or not coords_mapping[part_name]:
            # Create default grid when no specific coordinates available
            grid_shape = (5, 6, 4)  # DexH13 default grid dimensions
            x_range = (-0.01, 0.01)
            y_range = (-0.012, 0.012)
            z_range = (0.0, 0.02)

            x = torch.linspace(
                x_range[0], x_range[1], grid_shape[0], device=self.device
            )
            y = torch.linspace(
                y_range[0], y_range[1], grid_shape[1], device=self.device
            )
            z = torch.linspace(
                z_range[0], z_range[1], grid_shape[2], device=self.device
            )

            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
            points_tensor = torch.stack(
                [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1
            )
            return points_tensor, None

        # Process actual coordinate data with scaling
        coords = coords_mapping[part_name]
        scale_factor = 0.001  # Convert millimeters to meters
        points = [
            [
                coords[0][i] * scale_factor,
                coords[1][i] * scale_factor,
                coords[2][i] * scale_factor,
            ]
            for i in range(len(coords[0]))
        ]

        points_tensor = torch.tensor(points, device=self.device)

        # Apply DexH13-specific rotations for proper sensor orientation
        if part_name in self.x_rotation_parts:
            points_tensor = self._apply_rotation_x(
                points_tensor, self.x_rotation_parts[part_name]
            )
        if part_name in self.y_rotation_parts:
            points_tensor = self._apply_rotation_y(
                points_tensor, self.y_rotation_parts[part_name]
            )
        if part_name in self.z_rotation_parts:
            points_tensor = self._apply_rotation_z(
                points_tensor, self.z_rotation_parts[part_name]
            )

        return points_tensor, coords

    def get_grid_points(self, part_name: str) -> torch.Tensor:
        """
        Get grid points for specified part

        Args:
            part_name (str): Name of the finger part

        Returns:
            torch.Tensor: Grid points for the specified part
        """
        if part_name in self.custom_grid_points:
            return self.custom_grid_points[part_name]
        transformed_points, _ = self.create_default_grid(part_name)
        return transformed_points

    def _apply_rotation_x(self, points, angle_degrees):
        """
        Apply X-axis rotation to points

        Args:
            points (torch.Tensor): Points to rotate
            angle_degrees (float): Rotation angle in degrees

        Returns:
            torch.Tensor: Rotated points
        """
        import math

        angle_rad = math.radians(angle_degrees)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        rot_x = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]],
            device=self.device,
        )
        return torch.matmul(points, rot_x.T)

    def _apply_rotation_y(self, points, angle_degrees):
        """
        Apply Y-axis rotation to points

        Args:
            points (torch.Tensor): Points to rotate
            angle_degrees (float): Rotation angle in degrees

        Returns:
            torch.Tensor: Rotated points
        """
        import math

        angle_rad = math.radians(angle_degrees)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        rot_y = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]],
            device=self.device,
        )
        return torch.matmul(points, rot_y.T)

    def _apply_rotation_z(self, points, angle_degrees):
        """
        Apply Z-axis rotation to points

        Args:
            points (torch.Tensor): Points to rotate
            angle_degrees (float): Rotation angle in degrees

        Returns:
            torch.Tensor: Rotated points
        """
        import math

        angle_rad = math.radians(angle_degrees)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        rot_z = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
            device=self.device,
        )
        return torch.matmul(points, rot_z.T)


class DexH13JointController(IJointController):
    """
    DexH13 Joint Controller Adapter

    Manages joint mapping and motion generation for DexH13 robotic hand.
    Compatible with Tora robotic hand interface and provides conversion
    between degrees and radians for joint positions.
    """

    def __init__(self):
        """Initialize DexH13 joint controller with Tora-compatible mapping"""
        # DexH13 joint mapping compatible with Tora interface (13 joints)
        self.joint_mapping = {
            "RIGHT_THUMB_CMC_ABDUCTION": 9,  # Thumb abduction
            "RIGHT_THUMB_CMC_FLEXION": 10,  # Thumb flexion
            "RIGHT_THUMB_MCP_FLEXION": 11,  # Thumb MCP
            "RIGHT_THUMB_IP_FLEXION": 12,  # Thumb IP
            "RIGHT_INDEX_MCP_ABDUCTION": 0,  # Index abduction
            "RIGHT_INDEX_MCP_FLEXION": 1,  # Index MCP
            "RIGHT_INDEX_PIP_FLEXION": 2,  # Index PIP
            "RIGHT_MIDDLE_MCP_ABDUCTION": 3,  # Middle abduction
            "RIGHT_MIDDLE_MCP_FLEXION": 4,  # Middle MCP
            "RIGHT_MIDDLE_PIP_FLEXION": 5,  # Middle PIP
            "RIGHT_RING_MCP_ABDUCTION": 6,  # Ring abduction
            "RIGHT_RING_MCP_FLEXION": 7,  # Ring MCP
            "RIGHT_RING_PIP_FLEXION": 8,  # Ring PIP
        }

    def generate_joint_motion(self, positions: List[float]) -> torch.Tensor:
        """
        Generate DexH13 joint motion data compatible with Isaac Sim

        Args:
            positions (List[float]): Joint positions in radians

        Returns:
            torch.Tensor: Joint positions as a tensor for Isaac Sim
        """
        # positions_rad = np.deg2rad(positions)  # Convert degrees to radians
        # 需要确认关节角度单位为弧度，当前从tactile_sender_DexH13发送的mock数据本身为弧度，不需要转换
        positions_rad = positions

        joint_data = torch.zeros(
            16, dtype=torch.float32
        )  # DexH13 有16个关节 (根据实际关节数量调整)
        # 遍历 joint_mapping 中的关节名称和对应索引
        for joint_key, index in self.joint_mapping.items():
            if index < len(positions_rad):
                joint_data[index] = positions_rad[index]
        return joint_data

    def get_joint_mapping(self) -> Dict[str, int]:
        """
        Get joint mapping configuration

        Returns:
            Dict[str, int]: Joint name to index mapping
        """
        return self.joint_mapping


class DexH13MockDriver(IDeviceDriver):
    """
    DexH13 Mock Driver for Testing

    Provides a mock implementation of device driver interface
    for testing and simulation purposes. Returns simulated
    data without requiring actual hardware connection.
    """

    def __init__(self):
        """Initialize mock driver with connection state"""
        self.connected = False

    def connect(self) -> bool:
        """
        Simulate device connection

        Returns:
            bool: Always True for mock driver
        """
        self.connected = True
        return True

    def disconnect(self):
        """Simulate device disconnection"""
        self.connected = False

    def read_sensor_data(self) -> Optional[Dict]:
        """
        Read simulated sensor data

        Returns:
            Optional[Dict]: Mock sensor data or None if not connected
        """
        if not self.connected:
            return None
        # Return mock data for testing
        return {"tactile_data": {}, "joints": [0.0] * 13}

    def read_encoder_data(self) -> Optional[Dict]:
        """
        Read simulated encoder data

        Returns:
            Optional[Dict]: Mock encoder data or None if not connected
        """
        if not self.connected:
            return None
        return {"encoder_data": {}}
