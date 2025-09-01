"""
device_factory.py

Device Factory Implementation

This module implements the Factory design pattern for creating device-specific
components. It provides a centralized registry and factory for instantiating
different device adapters based on device type, promoting loose coupling and
extensibility.

Key Features:
- Factory pattern implementation for device component creation
- Type-safe component registration and instantiation
- Support for multiple device types with unified interface
- Extensible design for adding new device implementations

"""

from typing import Dict, Type
from px_replay.dataprocess.tactile.device_interfaces import (
    DeviceType,
    IDataReceiver,
    IGridManager,
    IJointController,
    IDeviceDriver,
)


class DeviceFactory:
    """
    Device Factory Class

    Implements the Factory design pattern for creating device-specific components.
    Maintains registries of implementation classes and provides factory methods
    for instantiating components based on device type.
    """

    def __init__(self):
        """Initialize factory with empty component registries"""
        # Registry for different device implementations
        self._data_receivers: Dict[DeviceType, Type[IDataReceiver]] = {}
        self._grid_managers: Dict[DeviceType, Type[IGridManager]] = {}
        self._joint_controllers: Dict[DeviceType, Type[IJointController]] = {}
        self._device_drivers: Dict[DeviceType, Type[IDeviceDriver]] = {}

    def register_data_receiver(
        self, device_type: DeviceType, receiver_class: Type[IDataReceiver]
    ):
        """
        Register data receiver implementation for specific device type

        Args:
            device_type (DeviceType): Target device type
            receiver_class (Type[IDataReceiver]): Implementation class
        """
        self._data_receivers[device_type] = receiver_class

    def register_grid_manager(
        self, device_type: DeviceType, manager_class: Type[IGridManager]
    ):
        """
        Register grid manager implementation for specific device type

        Args:
            device_type (DeviceType): Target device type
            manager_class (Type[IGridManager]): Implementation class
        """
        self._grid_managers[device_type] = manager_class

    def register_joint_controller(
        self, device_type: DeviceType, controller_class: Type[IJointController]
    ):
        """
        Register joint controller implementation for specific device type

        Args:
            device_type (DeviceType): Target device type
            controller_class (Type[IJointController]): Implementation class
        """
        self._joint_controllers[device_type] = controller_class

    def register_device_driver(
        self, device_type: DeviceType, driver_class: Type[IDeviceDriver]
    ):
        """
        Register device driver implementation for specific device type

        Args:
            device_type (DeviceType): Target device type
            driver_class (Type[IDeviceDriver]): Implementation class
        """
        self._device_drivers[device_type] = driver_class

    def create_data_receiver(self, device_type: DeviceType, **kwargs) -> IDataReceiver:
        """
        Create data receiver instance for specified device type

        Args:
            device_type (DeviceType): Target device type
            **kwargs: Constructor arguments for the receiver

        Returns:
            IDataReceiver: Data receiver instance

        Raises:
            ValueError: If device type is not registered
        """
        if device_type not in self._data_receivers:
            raise ValueError(f"Unregistered device type: {device_type}")
        return self._data_receivers[device_type](**kwargs)

    def create_grid_manager(self, device_type: DeviceType, **kwargs) -> IGridManager:
        """
        Create grid manager instance for specified device type

        Args:
            device_type (DeviceType): Target device type
            **kwargs: Constructor arguments for the manager

        Returns:
            IGridManager: Grid manager instance

        Raises:
            ValueError: If device type is not registered
        """
        if device_type not in self._grid_managers:
            raise ValueError(f"Unregistered device type: {device_type}")
        return self._grid_managers[device_type](**kwargs)

    def create_joint_controller(
        self, device_type: DeviceType, **kwargs
    ) -> IJointController:
        """
        Create joint controller instance for specified device type

        Args:
            device_type (DeviceType): Target device type
            **kwargs: Constructor arguments for the controller

        Returns:
            IJointController: Joint controller instance

        Raises:
            ValueError: If device type is not registered
        """
        if device_type not in self._joint_controllers:
            raise ValueError(f"Unregistered device type: {device_type}")
        return self._joint_controllers[device_type](**kwargs)

    def create_device_driver(self, device_type: DeviceType, **kwargs) -> IDeviceDriver:
        """
        Create device driver instance for specified device type

        Args:
            device_type (DeviceType): Target device type
            **kwargs: Constructor arguments for the driver

        Returns:
            IDeviceDriver: Device driver instance

        Raises:
            ValueError: If device type is not registered
        """
        if device_type not in self._device_drivers:
            raise ValueError(f"Unregistered device type: {device_type}")
        return self._device_drivers[device_type](**kwargs)


# Global factory instance for application-wide use
device_factory = DeviceFactory()
