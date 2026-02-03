#!/usr/bin/env python
import time
import sys
import os
import cv2
import numpy as np
from pynput import keyboard
import argparse

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type=int, default=0)
parser.add_argument('--folder', type=str, default='train')
args = parser.parse_args()

data_dir = os.path.join(script_path, "..", "data", args.folder)
if not os.path.exists(data_dir):
    print(f'Folder "{args.folder}" in path {data_dir} does not exist. Please create it.')
    sys.exit(1)

# --------------------------
# Robot init
# --------------------------
bot = PiBot(ip=args.ip)
bot.setVelocity(0, 0)

print("Get ready...")
for x in ["3", "2", "1", "GO!"]:
    print(x)
    time.sleep(1)

# --------------------------
# Control parameters (tune)
# --------------------------
TURN = 0.25          # gentle turn angle
SHARP = 0.50         # sharp turn angle

ALPHA = 0.20         # steering smoothing: 0.1 smoother, 0.3 more responsive
Ka = 15              # turning gain

BASE_SPEED = 15      # speed when straight
TURN_SPEED = 10      # speed when turning

SAVE_DT = 0.10       # seconds (10 Hz)

# --------------------------
# State variables
# --------------------------
steer_cmd = 0.0      # desired steering from keys
angle = 0.0          # smoothed steering applied
im_number = args.im_num
continue_running = True

# --------------------------
# Keyboard handlers
# --------------------------
def on_press(key):
    global steer_cmd, continue_running
    try:
        if key == keyboard.Key.up:
            steer_cmd = 0.0
            print("straight")
        elif key == keyboard.Key.left:
            steer_cmd = -TURN
            print("left")
        elif key == keyboard.Key.right:
            steer_cmd = TURN
            print("right")
        elif key == keyboard.Key.down:
            steer_cmd = 0.0
        elif key == keyboard.Key.space:
            print("stop")
            continue_running = False
            bot.setVelocity(0, 0)
    except Exception as e:
        print(f"An error occurred: {e}")
        continue_running = False
        bot.setVelocity(0, 0)

def on_release(key):
    global steer_cmd
    # when you release left/right, go back to straight
    if key in [keyboard.Key.left, keyboard.Key.right]:
        steer_cmd = 0.0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# --------------------------
# Main loop
# --------------------------
try:
    while continue_running:
        img = bot.getImage()
        if img is None:
            continue

        # 1) smooth steering (removes jerks)
        angle = (1.0 - ALPHA) * angle + ALPHA * steer_cmd

        # 2) snap to discrete levels (great for 5-class classification)
        if angle > 0.35:
            angle = SHARP
        elif angle > 0.10:
            angle = TURN
        elif angle < -0.35:
            angle = -SHARP
        elif angle < -0.10:
            angle = -TURN
        else:
            angle = 0.0

        angle = float(np.clip(angle, -0.5, 0.5))

        # 3) slow down while turning
        Kd = TURN_SPEED if abs(angle) > 0.05 else BASE_SPEED

        left = int(Kd + Ka * angle)
        right = int(Kd - Ka * angle)

        bot.setVelocity(left, right)

        # 4) save image with label in filename
        fname = f"{im_number:06d}{angle:+.2f}.jpg"
        cv2.imwrite(os.path.join(data_dir, fname), img)
        im_number += 1

        time.sleep(SAVE_DT)

finally:
    bot.setVelocity(0, 0)
    listener.stop()
    print("Script ended")


'''
import time
import sys
import os
import cv2
import numpy as np
from pynput import keyboard
import argparse

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--im_num', type = int, default = 0)
parser.add_argument('--folder', type = str, default = 'train')
args = parser.parse_args()

if not os.path.exists(script_path+"/../data/"+args.folder):
    data_path = script_path.replace('scripts', 'data')
    print(f'Folder "{args.folder}" in path {data_path} does not exist. Please create it.')
    exit()

bot = PiBot(ip=args.ip)
# stop the robot

bot.setVelocity(0, 0)

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")


# Initialize variables
angle = 0
im_number = args.im_num
continue_running = True

def on_press(key):
    global angle, continue_running
    try:
        if key == keyboard.Key.up:
            angle = 0
            print("straight")
        elif key == keyboard.Key.down:
            angle = 0
        elif key == keyboard.Key.right:
            print("right")
            angle += 0.1
        elif key == keyboard.Key.left:
            print("left")
            angle -= 0.1
        elif key == keyboard.Key.space:
            print("stop")
            bot.setVelocity(0, 0)
            continue_running = False
            # return False  # Stop listener

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.setVelocity(0, 0)

# Start the listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    while continue_running:
        # Get an image from the robot
        img = bot.getImage()
        
        angle = np.clip(angle, -0.5, 0.5)
        Kd = 15  # Base wheel speeds
        Ka = 15  # Turn speed
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
        
        bot.setVelocity(left, right)

        cv2.imwrite(script_path+"/../data/"+args.folder+"/"+str(im_number).zfill(6)+'%.2f'%angle+".jpg", img) 
        im_number += 1

        time.sleep(0.1)  # Small delay to reduce CPU usage

    # Clean up
    bot.setVelocity(0, 0)
    listener.stop()
    print("Script ended")


except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    listener.stop()
'''


