"""
UDP Sender for Robot Communication

This module provides a unified UDPSender class for transmitting
3D coordinates and trajectory predictions to a robot via UDP.

Used by:
- ball_tracker_params.py
- trajectory_predictor.py
- long_trajectory_predictor.py
"""

import socket
import struct
import time
from typing import Optional
import numpy as np


# Default robot configuration
DEFAULT_ROBOT_IP = "192.168.0.51"
DEFAULT_ROBOT_PORT = 5005


class UDPSender:
    """
    UDP sender for transmitting coordinates and predictions to robot.
    
    Supports:
    - Sending 3D coordinates (x, y, z)
    - Sending landing predictions (x, y, z, time_to_land)
    - Optional rate limiting
    - Enable/disable toggle
    """
    
    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT, 
                 rate_limit: Optional[float] = None):
        """
        Initialize UDP sender.
        
        Args:
            robot_ip: IP address of the robot
            port: UDP port number
            rate_limit: Optional rate limit in Hz (e.g., 30.0 for 30 Hz)
        """
        self.robot_ip = robot_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enabled = True
        self.packets_sent = 0
        
        # Rate limiting
        self.rate_limit = rate_limit
        self.last_send_time = 0
        self.min_send_interval = (1.0 / rate_limit) if rate_limit else 0
        
    def send_coordinates(self, x: float, y: float, z: float) -> bool:
        """
        Send 3D coordinates as packed floats (12 bytes: 3 floats).
        
        Args:
            x, y, z: 3D coordinates in meters
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        # Rate limiting check
        if self.rate_limit:
            current_time = time.time()
            if (current_time - self.last_send_time) < self.min_send_interval:
                return False
            self.last_send_time = current_time
        
        try:
            # Pack 3 floats into binary data
            packet = struct.pack('3f', x, y, z)
            self.sock.sendto(packet, (self.robot_ip, self.port))
            self.packets_sent += 1
            return True
        except Exception as e:
            print(f"[UDP ERROR] Failed to send: {e}")
            return False
    
    def send_landing_position(self, landing_pos: np.ndarray, time_to_land: float) -> bool:
        """
        Send landing position with time to landing (16 bytes: 4 floats x, y, z, t).
        
        Args:
            landing_pos: NumPy array with [x, y, z] landing position in meters
            time_to_land: Time to landing in seconds
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Pack 4 floats: x, y, z, time
            packet = struct.pack('4f', landing_pos[0], landing_pos[1], 
                               landing_pos[2], time_to_land)
            self.sock.sendto(packet, (self.robot_ip, self.port))
            self.packets_sent += 1
            return True
        except Exception as e:
            print(f"[UDP ERROR] Failed to send: {e}")
            return False
    
    def toggle(self) -> bool:
        """
        Toggle UDP sending on/off.
        
        Returns:
            Current enabled state
        """
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        print(f"[UDP] Sending {status}")
        return self.enabled
    
    def close(self):
        """Close the UDP socket."""
        self.sock.close()
