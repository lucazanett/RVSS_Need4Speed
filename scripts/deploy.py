#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from preprocess import PreProcessImage
# Setup Paths
from detector import StopSignDetector, StopSignController
from net_utils import get_transform
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot
from network import Net

# --- 2. CONFIGURATION ---
parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--model', type=str, default='steer_net_yuv.pth', help='Path to model file')
parser.add_argument('--is_regressor',type=bool,default=False)
parser.add_argument('--throttle_control',type=bool,default=False)
args = parser.parse_args()

bot = PiBot(ip=args.ip)
bot.setVelocity(0, 0)

# --- 3. LOAD NETWORK ---
print(f"Loading model from {args.model}...")
device = torch.device('cpu') # Robot usually runs on CPU
num_outputs=1 if args.is_regressor else 5
print("is regressor," ,args.is_regressor)


model = Net(num_outputs=num_outputs).to(device)
try:
    # Load weights (map_location ensures it works even if trained on GPU)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval() # Set to evaluation mode (disable dropout, etc)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- 4. DEFINE TRANSFORMS (Must match training) ---
# Note: We assume the robot camera gives a Numpy array. 
# ToTensor() converts (H, W, C) -> (C, H, W) and scales 0-255 to 0.0-1.0
preprocess = get_transform()
# --- 5. CLASS MAPPING ---
# You need to map the output Class ID (0, 1, 2...) to a Steering Angle.
# ADJUST THESE VALUES based on your training labels!
# Example assumption: 5 classes -> [Hard Left, Left, Straight, Right, Hard Right]
class_to_angle = {
    0: -0.5, # Hard Left
    1: -0.2, # Slight Left
    2:  0.0, # Straight
    3:  0.2, # Slight Right
    4:  0.5  # Hard Right
}

# Countdown
print("Get ready...")
time.sleep(1)
print("GO!")
print("throttle control is ", args.throttle_control)
prev_angle = 0.0
steps = 0
detector = StopSignDetector()
controller_stopper= StopSignController(1600)

try:
    while True:
        steps+=1
        if steps % 2 == 0:
            continue
        # 1. Get image from robot
        im_np = bot.getImage()[120:, :, :]
        im_to_net = im_np.copy()

        

        im_to_detector = cv2.cvtColor(im_np.copy(),cv2.COLOR_BGR2RGB)
        det = detector.detect(im_to_detector.copy())
        img_detection = detector.image_show(im_np.copy(),det)

        # Display both the raw image and detection image
        cv2.imshow("Raw Image", im_np)
        cv2.imshow("PiBot Autonomous View", img_detection)
        
        # Required to refresh the windows (1ms wait, non-blocking)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ctrl = controller_stopper.update(det)

        if ctrl["override"] is True:
            bot.setVelocity(0, 0)
            continue








        
        # 2. Preprocess
        # PiBot image is usually a Numpy array (H,W,C). 
        # Transform expects a PIL Image or Numpy array.
        input_tensor = preprocess(im_to_net)
        
        # Add batch dimension: [3, 40, 60] -> [1, 3, 40, 60]
        input_tensor = input_tensor.unsqueeze(0)

        # 3. Inference
        with torch.no_grad():
            output = model(input_tensor)
            
            # Get the class with the highest score
            # prediction is the index (e.g., 2)
            _, prediction = torch.max(output, 1)
            class_id = prediction.item()

        # 4. Map to Angle
        if args.is_regressor == False:
            angle = class_to_angle.get(class_id, 0.0)
        else:
            angle = class_id # Default to 0 if unknown class
        # delta = angle - prev_angle
        # delta = np.clip(delta,-2,2)
        # angle = prev_angle + delta
        # prev_angle = angle
        print(f"Pred Class: {class_id} | Angle: {angle}")
        
        # 5. Control Logic
        Kd = 15# Base speed 
        Ka = 15 # Turning aggressiveness
        if args.throttle_control:
            Kd = 10 if angle!= 0 else 30
        
        # Optional: Slow down for hard turns (Simple Dynamic Throttle)
        # if abs(angle) > 0.3:
        #     Kd = 10 # Slow down for corners
        
        left  = int(Kd + Ka*angle) 
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
        
        # Simple sleep to match camera rate (~10-20fps)
        time.sleep(0.05) 
            
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    print("\nStopping robot.")