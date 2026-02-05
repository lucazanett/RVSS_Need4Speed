#!/usr/bin/env python3
import time
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot


parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--model', type=str, default='steer_net.pth', help='Path to trained model weights')
parser.add_argument('--speed', type=float, default=20.0, help='Base speed (Kd)')
parser.add_argument('--turn_speed', type=float, default=20.0, help='Turn speed (Ka)')
parser.add_argument('--visualize', action='store_true', help='Show camera view with predictions')
args = parser.parse_args()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize robot
bot = PiBot(ip=args.ip)
bot.setVelocity(0, 0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)

        self.relu = nn.ReLU()


    def forward(self, x):
        #extract features with convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        #linear layer for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
       
        return x# Initialize network
net = Net().to(device)

# Load trained weights
if not os.path.exists(args.model):
    print(f"Error: Model file '{args.model}' not found!")
    sys.exit(1)

net.load_state_dict(torch.load(args.model, map_location=device))
net.eval()  # Set to evaluation mode
print(f"Loaded model from {args.model}")

# Define the same transform as in training (without data augmentation)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((40, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# Class labels (must match training)
class_labels = ['sharp left', 'left', 'straight', 'right', 'sharp right']

# Mapping from class index to steering angle
def class_to_angle(class_idx):
    """Convert predicted class to steering angle"""
    if class_idx == 0:  # sharp left
        return -0.5
    elif class_idx == 1:  # left
        return -0.25
    elif class_idx == 2:  # straight
        return 0.0
    elif class_idx == 3:  # right
        return 0.25
    elif class_idx == 4:  # sharp right
        return 0.5
    else:
        return 0.0

print("\nSteering Control:")
print("  Class 0 (sharp left)  -> angle = -0.5")
print("  Class 1 (left)        -> angle = -0.25")
print("  Class 2 (straight)    -> angle =  0.0")
print("  Class 3 (right)       -> angle =  0.25")
print("  Class 4 (sharp right) -> angle =  0.5")
print(f"\nBase Speed (Kd): {args.speed}")
print(f"Turn Speed (Ka): {args.turn_speed}")
print("\nPress Ctrl+C to stop the robot\n")

# Countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

#######################################################################################################################################
####     MAIN CONTROL LOOP                                                                                                         ####
#######################################################################################################################################

try:
    frame_count = 0
    total_time = 0
    
    while True:
        start_time = time.time()
        
        # Get image from robot
        img = bot.getImage()
        
        # Crop image (same as training - remove top 120 pixels)
        img_cropped = img[120:, :, :]
        
        # Apply transforms
        img_tensor = transform(img_cropped)
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get prediction from network
        with torch.no_grad():
            outputs = net(img_tensor)
            _, predicted_class = torch.max(outputs, 1)
            predicted_class = predicted_class.item()
        
        # Convert prediction to steering angle
        angle = class_to_angle(predicted_class)
        
        # Calculate wheel speeds
        Kd = args.speed      # base wheel speeds
        Ka = args.turn_speed # turn speed
        left  = int(Kd + Ka * angle)
        right = int(Kd - Ka * angle)
        
        # Set robot velocity
        bot.setVelocity(left, right)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        total_time += elapsed
        avg_fps = frame_count / total_time
        
        # Print status
        print(f"Frame {frame_count:04d} | Class: {predicted_class} ({class_labels[predicted_class]:11s}) | "
              f"Angle: {angle:+.2f} | L/R: {left:3d}/{right:3d} | FPS: {1/elapsed:.1f} (avg: {avg_fps:.1f})")
        
        # Visualize if requested
        if args.visualize:
            display_img = img.copy()
            
            # Add prediction text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Pred: {class_labels[predicted_class]} ({angle:+.2f})"
            cv2.putText(display_img, text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add speed text
            speed_text = f"L/R: {left}/{right}"
            cv2.putText(display_img, speed_text, (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw crop line
            cv2.line(display_img, (0, 120), (display_img.shape[1], 120), (255, 0, 0), 2)
            
            cv2.imshow("PiBot Autonomous View", display_img)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Small delay to prevent overwhelming the robot
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\n\nStopping robot...")
    bot.setVelocity(0, 0)
    if args.visualize:
        cv2.destroyAllWindows()
    print("Done!")

except Exception as e:
    print(f"\n\nError occurred: {e}")
    bot.setVelocity(0, 0)
    if args.visualize:
        cv2.destroyAllWindows()
    raise
