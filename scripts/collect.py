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
parser.add_argument('--im_num', type = int, default = 0)
parser.add_argument('--folder', type = str, default = 'train_5')
args = parser.parse_args()

if not os.path.exists(script_path+"/../data/"+args.folder):
    data_path = script_path.replace('scripts', 'data')
    print(f'Folder "{args.folder}" in path {data_path} does not exist. Please create it.')
    exit()
else:
    filenames = os.listdir(script_path+"/../data/"+args.folder)
    if (len(filenames)) != 0:
        
        filenames = [os.path.split(f)[-1].split(".jpg")[0][:6:] for f in filenames]
        print(script_path+"/../data/"+args.folder)
        filenames = sorted(filenames)
        im_number_files = filenames[-1]
    else:
        im_number_files = None

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
if args.im_num == 0 and im_number_files is not None:
    im_number = int(im_number_files)
continue_running = True
def on_press(key):
    global angle, continue_running, record
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (20, 50)       # Coordinates: (x, y) from top-left
        fontScale = 1
        color = (0, 255, 0)  # Green in BGR
        thickness = 2
        text = f"Angle: {angle:.2f}"
        display_img = img.copy()
        cv2.putText(display_img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.imshow("PiBot View", display_img)

        # 2. Define text properties

        # # This is required to refresh the image window
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     continue_running = False
        # --------------------------

        


        print("CURRENT ANGLE : ", angle)
        Kd = 15 # Base wheel speeds
        Ka = 15  # Turn speed
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
        
        bot.setVelocity(left, right)

        cv2.imwrite(script_path+"/../data/"+args.folder+"/"+str(im_number).zfill(6)+'%.2f'%angle+".jpg", img) 
        cv2.imwrite(script_path+"/../data/"+args.folder+"_displayed"+"/"+str(im_number).zfill(6)+'%.2f'%angle+".jpg", display_img)
        im_number += 1

        time.sleep(0.1)  # Small delay to reduce CPU usage

    # Clean up
    bot.setVelocity(0, 0)
    listener.stop()
    print("Script ended")


except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    listener.stop()