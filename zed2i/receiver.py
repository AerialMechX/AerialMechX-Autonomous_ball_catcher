import socket
import struct
import math

# Listen on all available network interfaces
HOST = '0.0.0.0' 
PORT = 5006 # Ensure this matches your camera's target port!

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"Simple Receiver listening on port {PORT}...")
print("Expecting 6 floats (x, y, z, roll, pitch, yaw)...")

while True:
    try:
        # Buffer size must be at least 24 bytes (6 floats * 4 bytes)
        # Using 1024 is safe.
        data, addr = sock.recvfrom(1024) 
        
        # Unpack the binary data into 6 floats
        # '6f' matches the sender's struct.pack('6f', ...)
        x, y, z, roll, pitch, yaw = struct.unpack('6f', data)
        
        # Display the relevant navigation data
        # Assuming Z is depth (forward/back) and Yaw is Heading
        print(f"Received: X={x:.3f}, Z={z:.3f}, Pitch={pitch:.3f} Yaw={yaw:.3f}")
        
    except struct.error as e:
        print(f"Packet Error (Size mismatch?): {e}")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
        
    except Exception as e:
        print(f"Error: {e}")
