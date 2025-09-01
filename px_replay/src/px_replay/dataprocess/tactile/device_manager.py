"""
device_manager.py

Unified Device Manager Implementation

This module implements a unified device management system using the Strategy
pattern. It provides a single interface for managing different device types
while delegating specific operations to device-appropriate implementations.

Key Features:
- Strategy pattern for device-specific behavior
- Lazy initialization of components
- Centralized resource management and cleanup
- Type-safe device component access


"""

from typing import Optional, List
from px_replay.dataprocess.tactile.device_interfaces import (
    DeviceType,
    IDataReceiver,
    IGridManager,
    IJointController,
    IDeviceDriver,
)
from px_replay.dataprocess.tactile.device_factory import device_factory

from px_replay.dataprocess.tactile.adapters_Dexh13 import (
    DexH13DataReceiver,
    DexH13GridManager,
    DexH13JointController,
    DexH13MockDriver,
)
from px_replay.dataprocess.tactile.adapters_DataGlove import (
    DataGloveDataReceiver,
    DataGloveGridManager,
    DataGloveJointController,
    DataGloveSerialDriver,
)


class DeviceManager:
    """
    Unified Device Manager

    Implements the Strategy pattern to manage different device types through
    a unified interface. Provides lazy initialization, resource management,
    and type-safe component access for various robotic devices.
    """

    def __init__(self, device_type: DeviceType):
        """
        Initialize device manager for specific device type

        Args:
            device_type (DeviceType): Type of device to manage
        """
        self.device_type = device_type

        # Lazy-initialized components
        self._data_receiver: Optional[IDataReceiver] = None
        self._grid_manager: Optional[IGridManager] = None
        self._joint_controller: Optional[IJointController] = None
        self._device_driver: Optional[IDeviceDriver] = None

        # Register device implementations in factory
        self._register_implementations()

    def _register_implementations(self):
        """
        Register all device implementations in the factory

        Sets up the factory with concrete implementations for each
        supported device type, enabling polymorphic component creation.
        """
        # Register DexH13 device implementations
        device_factory.register_data_receiver(DeviceType.DEXH13, DexH13DataReceiver)
        device_factory.register_grid_manager(DeviceType.DEXH13, DexH13GridManager)
        device_factory.register_joint_controller(
            DeviceType.DEXH13, DexH13JointController
        )
        device_factory.register_device_driver(DeviceType.DEXH13, DexH13MockDriver)

        # Register DataGlove device implementations
        device_factory.register_data_receiver(
            DeviceType.DATAGLOVE, DataGloveDataReceiver
        )
        device_factory.register_grid_manager(DeviceType.DATAGLOVE, DataGloveGridManager)
        device_factory.register_joint_controller(
            DeviceType.DATAGLOVE, DataGloveJointController
        )
        device_factory.register_device_driver(
            DeviceType.DATAGLOVE, DataGloveSerialDriver
        )

    def get_data_receiver(self, **kwargs) -> IDataReceiver:
        """
        Get data receiver component with lazy initialization

        Args:
            **kwargs: Runtime parameters for receiver initialization

        Returns:
            IDataReceiver: Device-specific data receiver instance
        """
        if self._data_receiver is None:
            self._data_receiver = device_factory.create_data_receiver(
                self.device_type, **kwargs
            )
        return self._data_receiver

    def get_grid_manager(self, **kwargs) -> IGridManager:
        """
        Get grid manager component with lazy initialization

        Args:
            **kwargs: Runtime parameters for manager initialization

        Returns:
            IGridManager: Device-specific grid manager instance
        """
        if self._grid_manager is None:
            self._grid_manager = device_factory.create_grid_manager(
                self.device_type, **kwargs
            )
        return self._grid_manager

    def get_joint_controller(self, **kwargs) -> IJointController:
        """
        Get joint controller component with lazy initialization

        Args:
            **kwargs: Runtime parameters for controller initialization

        Returns:
            IJointController: Device-specific joint controller instance
        """
        if self._joint_controller is None:
            self._joint_controller = device_factory.create_joint_controller(
                self.device_type, **kwargs
            )
        return self._joint_controller

    def get_device_driver(self, **kwargs) -> IDeviceDriver:
        """
        Get device driver component with lazy initialization

        Args:
            **kwargs: Runtime parameters for driver initialization

        Returns:
            IDeviceDriver: Device-specific driver instance
        """
        if self._device_driver is None:
            self._device_driver = device_factory.create_device_driver(
                self.device_type, **kwargs
            )
        return self._device_driver

    def initialize_all_components(self, **kwargs):
        """
        Initialize all components for the device

        Performs lazy initialization of all device components with
        optional runtime parameters for each component type.

        Args:
            **kwargs: Component-specific initialization parameters
                - receiver_kwargs: Parameters for data receiver
                - grid_kwargs: Parameters for grid manager
                - joint_kwargs: Parameters for joint controller
                - driver_kwargs: Parameters for device driver
        """
        self.get_data_receiver(**kwargs.get("receiver_kwargs", {}))
        self.get_grid_manager(**kwargs.get("grid_kwargs", {}))
        self.get_joint_controller(**kwargs.get("joint_kwargs", {}))
        self.get_device_driver(**kwargs.get("driver_kwargs", {}))

    def cleanup(self):
        """
        Cleanup all resources and stop active components

        Gracefully stops data reception, disconnects devices,
        and releases all allocated resources to prevent memory leaks.
        """
        # Stop data receiver if active
        if self._data_receiver:
            try:
                self._data_receiver.stop_receiver()
            except Exception as e:
                print(f"Error stopping data receiver: {e}")

        # Disconnect device driver if connected
        if self._device_driver:
            try:
                self._device_driver.disconnect()
            except Exception as e:
                print(f"Error disconnecting device driver: {e}")

        # Reset component references
        self._data_receiver = None
        self._grid_manager = None
        self._joint_controller = None
        self._device_driver = None
