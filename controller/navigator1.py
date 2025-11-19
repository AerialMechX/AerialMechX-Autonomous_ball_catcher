import serial
import time
import math

ARDUINO_PORT = "/dev/ttyACM0" 
BAUD_RATE = 9600

TARGET_X = 5.0
TARGET_Y = 3.0

HEADING_TOLERANCE = 0.1
DISTANCE_TOLERANCE = 0.2

def get_position_from_camera():
    print("WARNING: Using FAKE camera data!")
    global fake_x, fake_y, angel
    fake_x += 0.01 * math.cos(angel)
    fake_y += 0.01 * math.sin(angel)
    if angel < math.pi/4:
         angel += 0.01
    return fake_x, fake_y, angel

fake_x, fake_y, angel = 0.0, 0.0, 0.0

try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2)

    while True:
        current_x, current_y, current_angle = get_position_from_camera()
        print(f"Current Pos: ({current_x:.2f}, {current_y:.2f}) @ {math.degrees(current_angle):.1f} deg")

        error_x = TARGET_X - current_x
        error_y = TARGET_Y - current_y
        distance_to_goal = math.sqrt(error_x**2 + error_y**2)

        if distance_to_goal < DISTANCE_TOLERANCE:
            print("GOAL REACHED!")
            arduino.write(b's')
            break

        angle_to_goal = math.atan2(error_y, error_x)
        
        heading_error = angle_to_goal - current_angle
        
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi
        
        print(f"Target: {math.degrees(angle_to_goal):.1f} deg | Error: {math.degrees(heading_error):.1f} deg | Dist: {distance_to_goal:.2f} m")

        if abs(heading_error) > HEADING_TOLERANCE:
            if heading_error > 0:
                print("Action: Turn Right ('d')")
                arduino.write(b'd')
            else:
                print("Action: Turn Left ('a')")
                arduino.write(b'a')
        else:
            print("Action: Move Forward ('w')")
            arduino.write(b'w')
            
        time.sleep(0.1)

except serial.SerialException as e:
    print(f"Error: Could not open port {ARDUINO_PORT}. {e}")
except KeyboardInterrupt:
    print("\nShutting down navigation...")
finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.write(b's')
        arduino.close()
        print("Serial port closed.")
