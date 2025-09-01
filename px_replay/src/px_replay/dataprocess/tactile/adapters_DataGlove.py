"""
adapters_DataGlove.py

DataGlove Device Adapter Implementation

This module implements the DataGlove-specific adapters that conform to the
device interface contracts. It provides concrete implementations for data
reception, grid management, joint control, and device communication for
DataGlove hardware.

Key Features:
- UDP-based data reception for DataGlove sensor data
- Grid point management with coordinate transformation
- Joint mapping and motion control for DataGlove joints
- Serial communication driver for hardware interface

"""

import json
import os
import torch
import serial
import struct
import numpy as np

from typing import Dict, List, Optional, Tuple

from px_replay.dataprocess.tactile.device_interfaces import (
    IDataReceiver,
    IGridManager,
    IJointController,
    IDeviceDriver,
)


class DataGloveDataReceiver(IDataReceiver):
    """
    DataGlove Data Receiver Adapter

    Implements UDP-based data reception for DataGlove device.
    Handles real-time sensor data and joint position information
    through network communication.
    """

    def __init__(self, host="127.0.0.1", port=12345):
        """
        Initialize DataGlove data receiver

        Args:
            host (str): UDP server host address
            port (int): UDP server port number
        """
        self.host = host
        self.port = port
        self.sock = None
        self.latest_data = None
        self.running = False

    def start_receiver(self):
        """
        Start UDP data reception

        Creates UDP socket and binds to specified host:port
        for receiving DataGlove sensor data.
        """
        import socket

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.running = True

    def get_latest_data(self) -> Optional[Dict]:
        """
        Get latest received data

        Returns:
            Optional[Dict]: Latest sensor data or None if no data available
        """
        if not self.running or not self.sock:
            return None

        try:
            data, _ = self.sock.recvfrom(36000)  # default:4096
            self.latest_data = json.loads(data.decode())
            return self.latest_data
        except:
            return self.latest_data

    def stop_receiver(self):
        """Stop data reception and close socket"""
        self.running = False
        if self.sock:
            self.sock.close()


class DataGloveGridManager(IGridManager):
    """
    DataGlove Grid Manager Adapter

    Manages tactile sensor grid points for DataGlove device.
    Handles coordinate loading, transformation, and grid generation
    specific to DataGlove's sensor layout.
    """

    def __init__(self, device=None):
        """
        Initialize DataGlove grid manager

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

        # DataGlove-specific configuration (currently empty, can be extended)
        self.x_rotation_parts = {}
        self.y_rotation_parts = {}
        self.z_rotation_parts = {}

    def load_coords(self) -> Dict:
        """
        Load DataGlove coordinate data from configuration file

        Returns:
            Dict: Coordinate data for all DataGlove parts
        """
        from px_replay.config.paths import ROBOT_PATH

        grid_coords_file = os.path.join(
            ROBOT_PATH, "DataGlove", "tactile_data", "grid_coords.json"
        )

        try:
            with open(grid_coords_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def create_default_grid(self, part_name: str) -> Tuple[torch.Tensor, List]:
        """
        Create default grid points for DataGlove parts

        Args:
            part_name (str): Name of the body part (e.g., 'J32L', 'M42L')

        Returns:
            Tuple[torch.Tensor, List]: Grid points tensor and original coordinates
        """
        # DataGlove-specific part mapping
        coords_mapping = {
            "J32L": self.coords_data.get("J32L", []),
            "J42L": self.coords_data.get("J42L", []),
            "M42L": self.coords_data.get("M42L", []),
            "M6L": self.coords_data.get("M6L", []),
            "S22L": self.coords_data.get("S22L", []),
            "S32L": self.coords_data.get("S32L", []),
            "S6L": self.coords_data.get("S6L", []),
            "W22L": self.coords_data.get("W22L", []),
            "W32L": self.coords_data.get("W32L", []),
            "W6L": self.coords_data.get("W6L", []),
            "X22L": self.coords_data.get("X22L", []),
            "X6L": self.coords_data.get("X6L", []),
            "Z22L": self.coords_data.get("Z22L", []),
            "Z32L": self.coords_data.get("Z32L", []),
            "Z6L": self.coords_data.get("Z6L", []),
        }

        if part_name not in coords_mapping or not coords_mapping[part_name]:
            # Create default grid when no specific coordinates available
            grid_shape = (4, 5, 3)  # DataGlove default grid dimensions
            x_range = (-0.008, 0.008)
            y_range = (-0.01, 0.01)
            z_range = (0.0, 0.015)

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

        # Process actual coordinate data
        coords = coords_mapping[part_name]
        scale_factor = 0.001  # Convert to meters
        points = [
            [
                coords[0][i] * scale_factor,
                coords[1][i] * scale_factor,
                coords[2][i] * scale_factor,
            ]
            for i in range(len(coords[0]))
        ]

        points_tensor = torch.tensor(points, device=self.device)
        return points_tensor, coords

    def get_grid_points(self, part_name: str) -> torch.Tensor:
        """
        Get grid points for specified part

        Args:
            part_name (str): Name of the body part

        Returns:
            torch.Tensor: Grid points for the specified part
        """
        if part_name in self.custom_grid_points:
            return self.custom_grid_points[part_name]
        transformed_points, _ = self.create_default_grid(part_name)
        return transformed_points


class DataGloveJointController(IJointController):
    """
    DataGlove Joint Controller Adapter

    Manages joint mapping and motion generation for DataGlove device.
    Handles the conversion between sensor data and joint positions
    according to DataGlove's joint configuration.
    """

    def __init__(self):
        """Initialize DataGlove joint controller with specific joint mapping"""
        # DataGlove-specific joint mapping (29 joints total)
        self.joint_mapping = {
            "M4J": 19,
            "M5J": 24,
            "S4J": 20,
            "S5J": 25,
            "S2J": 10,
            "S3J": 15,
            "Z4J": 23,
            "Z5J": 28,
            "Z2J": 13,
            "Z3J": 18,
            "J1J": 0,
            "J2J": 1,
            "W4J": 21,
            "W5J": 26,
            "W2J": 11,
            "W3J": 16,
            "X4J": 22,
            "X5J": 27,
            "X2J": 12,
            "X3J": 17,
            "J4J": 3,
            "J3J": 2,
            "S1J": 5,
            "M1J": 4,
            "W1J": 6,
            "X1J": 7,
            "Z1J": 8,
            "M2J": 9,
            "M3J": 14,
        }

    def generate_joint_motion(self, positions: List[float]) -> torch.Tensor:
        """
        Generate DataGlove joint motion data

        Args:
            positions (List[float]): Joint position values in radians

        Returns:
            torch.Tensor: Joint positions as a tensor for Isaac Sim
        """

        # positions_rad = np.deg2rad(positions)  # 转换为弧度
        # 需要确认关节角度单位为弧度，当前从tactile_sender_DataGlove发送的mock数据本身为弧度，不需要转换
        positions_rad = positions

        joint_data = torch.zeros(29, dtype=torch.float32)  # DataGlove 有29个关节
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


class DataGloveSerialDriver(IDeviceDriver):
    """
    DataGlove Serial Communication Driver

    Implements serial communication interface for DataGlove hardware.
    Handles low-level communication protocols including sensor data
    reading and encoder data acquisition.
    """

    def __init__(self, serial_port="/dev/ttyACM0", baudrate=115200):
        """
        Initialize DataGlove serial driver

        Args:
            serial_port (str): Serial port path
            baudrate (int): Communication baud rate
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False

    def connect(self) -> bool:
        """
        Connect to DataGlove device via serial port

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.serial_connection = serial.Serial(
                self.serial_port, self.baudrate, timeout=1
            )
            if self.serial_connection.is_open:
                self.is_connected = True
                return True
        except Exception as e:
            print(f"Connection failed: {e}")
        return False

    def disconnect(self):
        """Disconnect from DataGlove device"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.is_connected = False

    def read_sensor_data(self) -> Optional[Dict]:
        """
        Read tactile sensor data from DataGlove

        Returns:
            Optional[Dict]: Sensor data or None if read failed
        """
        if not self.is_connected:
            return None

        try:
            # Send sensor data request command
            cmd = struct.pack("<Q", 1)  # CMD_SENSOR_DATA = 1
            request = b"\x55\x55" + cmd[:4]
            self.serial_connection.write(request)

            response = self.serial_connection.read(3417)
            if len(response) == 3417:
                return self._parse_sensor_data(response)
        except Exception as e:
            print(f"Failed to read sensor data: {e}")
        return None

    def read_encoder_data(self) -> Optional[Dict]:
        """
        Read encoder data from DataGlove

        Returns:
            Optional[Dict]: Encoder data or None if read failed
        """
        if not self.is_connected:
            return None

        try:
            # Send encoder data request command
            cmd = struct.pack("<Q", 2)  # CMD_ENCODER_ERRCODE = 2
            request = b"\x55\x55" + cmd[:4]
            self.serial_connection.write(request)

            response = self.serial_connection.read(72)
            if len(response) == 72:
                return self._parse_encoder_data(response)
        except Exception as e:
            print(f"Failed to read encoder data: {e}")
        return None

    def _parse_sensor_data(self, data: bytes) -> Dict:
        """
        Parse raw sensor data bytes

        Args:
            data (bytes): Raw sensor data from device

        Returns:
            Dict: Parsed sensor data
        """
        # Implement specific sensor data parsing logic
        return {"sensors": {}}

    def _parse_encoder_data(self, data: bytes) -> Dict:
        """
        Parse raw encoder data bytes

        Args:
            data (bytes): Raw encoder data from device

        Returns:
            Dict: Parsed encoder data
        """
        # Implement specific encoder data parsing logic
        return {"mag": {}}
