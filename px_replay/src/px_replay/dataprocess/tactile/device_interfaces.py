"""
device_interfaces.py

Device Interface Definitions

This module defines the abstract interfaces and contracts for device components
using the Interface Segregation Principle. It provides a unified API for
different device types while maintaining loose coupling and high cohesion.

Key Features:
- Abstract base classes defining component interfaces
- Device type enumeration for type safety
- Consistent API across different device implementations
- Support for dependency inversion principle

"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from enum import Enum


class DeviceType(Enum):
    """
    Enumeration of supported device types

    Provides type-safe device identification and ensures
    consistent device type handling across the system.
    """

    DEXH13 = "Dexh13"
    DEXH5 = "Dexh5"
    DATAGLOVE = "DataGlove"
    DEXH13_LEAPMOTION = "Dexh13_Leapmotion"


class IDataReceiver(ABC):
    """
    Data Receiver Interface

    Abstract interface for data reception components following the
    Dependency Inversion Principle. Defines contract for receiving
    real-time data from various device sources.
    """

    @abstractmethod
    def get_latest_data(self) -> Optional[Dict]:
        """
        Get latest received data from device

        Returns:
            Optional[Dict]: Latest data dictionary or None if no data available
        """
        pass

    @abstractmethod
    def start_receiver(self):
        """
        Start data reception process

        Initializes and begins data reception from the device source.
        Implementation should handle connection setup and error recovery.
        """
        pass

    @abstractmethod
    def stop_receiver(self):
        """
        Stop data reception and cleanup resources

        Gracefully stops data reception and releases any allocated
        resources such as network connections or file handles.
        """
        pass


class IGridManager(ABC):
    """
    Grid Manager Interface

    Abstract interface for managing tactile sensor grid points.
    Handles coordinate loading, transformation, and grid generation
    for different device configurations.
    """

    @abstractmethod
    def get_grid_points(self, part_name: str) -> torch.Tensor:
        """
        Get grid points for specified body part

        Args:
            part_name (str): Name of the body part (e.g., 'szyd_s', 'J32L')

        Returns:
            torch.Tensor: 3D coordinates of grid points for the part
        """
        pass

    @abstractmethod
    def load_coords(self) -> Dict:
        """
        Load coordinate data from configuration files

        Returns:
            Dict: Coordinate data for all supported body parts
        """
        pass

    @abstractmethod
    def create_default_grid(self, part_name: str) -> Tuple[torch.Tensor, List]:
        """
        Create default grid points for specified part

        Args:
            part_name (str): Name of the body part

        Returns:
            Tuple[torch.Tensor, List]: Grid points tensor and original coordinates
        """
        pass


class IJointController(ABC):
    """
    Joint Controller Interface

    Abstract interface for joint control and motion generation.
    Handles joint mapping, position control, and motion planning
    for robotic devices.
    """

    @abstractmethod
    def generate_joint_motion(self, positions: List[float]) -> Any:
        """
        Generate joint motion data from position inputs

        Args:
            positions (List[float]): Joint position values

        Returns:
            Any: Device-specific joint motion data structure
        """
        pass

    @abstractmethod
    def get_joint_mapping(self) -> Dict[str, int]:
        """
        Get joint name to index mapping

        Returns:
            Dict[str, int]: Mapping of joint names to their indices
        """
        pass


class IDeviceDriver(ABC):
    """
    Device Driver Interface

    Abstract interface for low-level device communication.
    Handles hardware connection, data reading, and device
    control operations.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to physical device

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Disconnect from physical device

        Closes device connection and releases hardware resources.
        Should be called during cleanup to prevent resource leaks.
        """
        pass

    @abstractmethod
    def read_sensor_data(self) -> Optional[Any]:
        """
        Read sensor data from device

        Returns:
            Optional[Any]: Sensor data or None if read failed
        """
        pass

    @abstractmethod
    def read_encoder_data(self) -> Optional[Any]:
        """
        Read encoder data from device

        Returns:
            Optional[Any]: Encoder data or None if read failed
        """
        pass
