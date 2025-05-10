"""drone_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Keyboard # type: ignore
import socket
import math
import time

# create the Robot instance.
robot = Robot()
keyboard = Keyboard()

# connect to the server
server_ip = '127.0.0.1'
server_port = 12345

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))
client_socket.setblocking(False) # continue if no message received
client_socket.sendall("drone".encode("utf-8"))

def get_server_command():
    command = None
    while True:
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                command = data.strip()
        except BlockingIOError:
            break
    return command

def send_server(data):
    pass

def shutdown_client():
    print("Shutting down client")
    try:
        client_socket.shutdown(socket.SHUT_RDWR)  # Politely shutdown both read and write
    except:
        pass  # Ignore errors (already disconnected etc.)

    try:
        client_socket.close()  # Actually close socket
    except:
        pass

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

gps = robot.getDevice("gps")
gps.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
inertial_unit = robot.getDevice("inertial unit")
inertial_unit.enable(timestep)

# initialize propellers
fr_motor = robot.getDevice("front right propeller")
fl_motor = robot.getDevice("front left propeller")
rr_motor = robot.getDevice("rear right propeller")
rl_motor = robot.getDevice("rear left propeller")
motors = [fr_motor, fl_motor, rr_motor, rl_motor]

for i in motors:
    i.setPosition(float('inf'))
    i.setVelocity(1)

while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

# constants
# vert_thrust = 68.5
# vert_offset = 0.6
# vert_p = 3.0
# roll_p = 50.0
# pitch_p = 30.0

vert_thrust = 68.0
vert_offset = 0.4
Kp = 5.0
Kd = 3.0
vert_p = 5.0
roll_p = 35.0
pitch_p = 25.0

target_altitude = 1 # initial target altitude
process_interval = 0.25 # how frequently a command is processed from server


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def calculate_thrust(h_target, h_current, T_max, k):
    # calculate thrust using exponential decay function
    d = abs(h_target - h_current)
    thrust = T_max * math.exp(-k * d)
    return thrust if h_target > h_current else -thrust

prev_alt = gps.getValues()[2]
# last_command_time = 0
# command_mult = 1
commands = {"MOVE_FORWARD":0, "MOVE_BACK":0, "MOVE_UP":0, "MOVE_DOWN":0, "ROTATE_LEFT":0, "ROTATE_RIGHT":0}
# presses = 1
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    server_command = get_server_command()
    if server_command in commands:
        # presses += 1
        commands[server_command] += 0.05 # duration per received command 

    # get current position data from sensors
    current_pos = gps.getValues()
    roll, pitch, yaw = inertial_unit.getRollPitchYaw()
    roll_velo, pitch_velo, yaw_velo = gyro.getValues()

    # disturbances to current position data
    roll_dist = 0
    pitch_dist = 0
    yaw_dist = 0

    manual_yaw = False
    manual_pitch = False
    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.UP:
            pitch_dist = -2.0
            manual_pitch = True
        elif key == Keyboard.DOWN:
            pitch_dist = 2.0
            manual_pitch = True
        elif key == Keyboard.RIGHT:
            yaw_dist = -1.3
            manual_yaw = True
        elif key == Keyboard.LEFT:
            yaw_dist = 1.3
            manual_yaw = True
        elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
            roll_dist = -1.0
        elif key == (Keyboard.SHIFT + Keyboard.LEFT):
            roll_dist = 1.0
        elif key == (Keyboard.SHIFT + Keyboard.UP):
            target_altitude += 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
        elif key == (Keyboard.SHIFT + Keyboard.DOWN):
            target_altitude -= 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
        presses += 1
        key = keyboard.getKey()  # Read next key press

    if commands:
        if commands["MOVE_FORWARD"] > 0:
            pitch_dist -= 1.5
            manual_pitch = True
            commands["MOVE_FORWARD"] -= timestep / 1000.0 # TS in ms
            if commands["MOVE_FORWARD"] < 0:
                commands["MOVE_FORWARD"] = 0
        elif commands["MOVE_BACK"] > 0:
            pitch_dist += 1.5
            manual_pitch = True
            commands["MOVE_BACK"] -= timestep / 1000.0
            if commands["MOVE_BACK"] < 0:
                commands["MOVE_BACK"] = 0
        elif commands["MOVE_UP"] > 0:
            target_altitude += 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
            commands["MOVE_UP"] -= timestep / 1000.0
            if commands["MOVE_UP"] < 0:
                commands["MOVE_UP"] = 0
        elif commands["MOVE_DOWN"] > 0:
            target_altitude += 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
            commands["MOVE_DOWN"] -= timestep / 1000.0
            if commands["MOVE_DOWN"] < 0:
                commands["MOVE_DOWN"] = 0
        elif commands["ROTATE_RIGHT"] > 0:
            roll_dist = -1.3 * 0.5
            commands["ROTATE_RIGHT"] -= timestep / 1000.0
            if commands["ROTATE_RIGHT"] < 0:
                commands["ROTATE_RIGHT"] = 0
        elif commands["ROTATE_LEFT"] > 0:
            roll_dist = 1.3 * 0.5
            commands["ROTATE_LEFT"] -= timestep / 1000.0
            if commands["ROTATE_LEFT"] < 0:
                commands["ROTATE_LEFT"] = 0

    roll_input = roll_p * clamp(roll + 0.002, -1.0, 1.0) + roll_velo + roll_dist
    pitch_input = pitch_p * clamp(pitch + 0.0057, -1.0, 1.0) + pitch_velo + pitch_dist
    
    yaw_input = yaw_dist

    # # pitch stabilization
    # small_pitch_threshold = 0.001
    # pitch_correction = 20.0

    # if not manual_pitch:
    #     if abs(pitch) > small_pitch_threshold:
    #         pitch_input -= pitch_correction * pitch

    # yaw stabilization

    small_yaw_threshold = 0.001 # rads/sec
    yaw_correction = 5.0 # how aggressive to counter
    if not manual_yaw:
        if abs(yaw_velo) > small_yaw_threshold:
            yaw_input -= yaw_correction * yaw_velo

    altitude_error = target_altitude - current_pos[2] + vert_offset
    clamped_alt_diff = clamp(altitude_error, -0.5, 0.5)
    current_alt = gps.getValues()[2]
    vertical_velocity = (current_alt - prev_alt) / (timestep / 1000.0)

    vert_input = (Kp * clamped_alt_diff) - (Kd * vertical_velocity)
    vert_input = clamp(vert_input, -5.0, 5.0)

    prev_alt = current_alt

    fl_motor_input = vert_thrust + vert_input - roll_input + pitch_input - yaw_input
    fr_motor_input = vert_thrust + vert_input + roll_input + pitch_input + yaw_input
    rl_motor_input = vert_thrust + vert_input - roll_input - pitch_input + yaw_input
    rr_motor_input = vert_thrust + vert_input + roll_input - pitch_input - yaw_input

    fl_motor.setVelocity(fl_motor_input)
    fr_motor.setVelocity(-fr_motor_input)
    rl_motor.setVelocity(-rl_motor_input)
    rr_motor.setVelocity(rr_motor_input)
    
    
    

# Enter here exit cleanup code.
shutdown_client()

"""
# TODO:
# Change to pulse-based system, where current action is incremented by
# time rather than individual pulses
# remove rate limiter, implement queue or multiple pulses concurrently

"""