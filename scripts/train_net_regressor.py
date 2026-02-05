import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics
import argparse

import matplotlib.pyplot as plt
from torch.utils.data import random_split
from steerDS import SteerDataSet
from preprocess import PreProcessImage
from network import Net
from net_utils import get_transform
#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################
# torch.manual_seed(0)
print(torch.rand(1))
#Helper function for visualising images in our dataset
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:,:,::-1]
    plt.imshow(rgbimg)
    plt.show()

#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train steering angle regressor')
parser.add_argument('--data', type=str, default='train_5',
                    help='Name of the training data folder in data/ (default: train_5)')
parser.add_argument('--model', type=str, default='steer_regressor.pth',
                    help='Name for the saved model file (default: steer_regressor.pth)')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of training epochs (default: 30)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size for training (default: 8)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate (default: 0.001)')
args = parser.parse_args()

print(f"Training configuration:")
print(f"  Data folder: {args.data}")
print(f"  Model name: {args.model}")
print(f"  Epochs: {args.epochs}")
print(f"  Batch size: {args.batch_size}")
print(f"  Learning rate: {args.lr}")

# Check if model file already exists
if os.path.exists(args.model):
    print(f"\n⚠ WARNING: Model file '{args.model}' already exists!")
    while True:
        response = input("Do you want to (o)verride it or use a (d)ifferent name? [o/d]: ").strip().lower()
        if response == 'o':
            print(f"Will override existing model '{args.model}'")
            break
        elif response == 'd':
            new_name = input(f"Enter new model name (e.g., 'steer_net_v2.pth'): ").strip()
            if new_name:
                args.model = new_name
                print(f"Will save model as '{args.model}'")
                break
            else:
                print("Invalid name. Please try again.")
        else:
            print("Please enter 'o' for override or 'd' for different name.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUNNING ON ", device)
#transformations for raw images before going to CNN
transform = get_transform()


script_path = os.path.dirname(os.path.realpath(__file__))

###################
## Train dataset ##
###################
# 1. Load the FULL dataset first
full_ds = SteerDataSet(os.path.join(script_path, '..', 'data', args.data), '.jpg', transform,is_regressor=True)

# 2. Calculate the split sizes (80% Train, 20% Validation)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size

# 3. Create the random split
# This automatically shuffles and assigns images to one of the two sets
train_ds, val_ds = random_split(full_ds, [train_size, val_size])


#data loader nicely batches images for the training process and shuffles (if desired)
trainloader = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
all_y = []
for S in trainloader:
    im, y = S    
    all_y += y.tolist()


print(f'Input to network shape: {im.shape}')

# Visualize the distribution of steering angles (continuous values)
plt.figure(figsize=(10, 5))
plt.hist(all_y, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Steering Angle')
plt.ylabel('Count')
plt.title('Training Dataset - Steering Angle Distribution')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Center (0°)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f'Steering angle statistics:')
print(f'  Min: {np.min(all_y):.3f}')
print(f'  Max: {np.max(all_y):.3f}')
print(f'  Mean: {np.mean(all_y):.3f}')
print(f'  Std: {np.std(all_y):.3f}')

# visualise some images and print angles
example_ims, example_angles = next(iter(trainloader))
print('Sample angles:', ' '.join(f'{example_angles[j]:.2f}' for j in range(min(8, len(example_angles)))))
imshow(torchvision.utils.make_grid(example_ims[:8]))


########################
## Validation dataset ##
########################


#data loader nicely batches images for the training process and shuffles (if desired)
valloader = DataLoader(val_ds,batch_size=1)
all_y = []
for S in valloader:
    im, y = S    
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

# Visualize the distribution of validation steering angles
plt.figure(figsize=(10, 5))
plt.hist(all_y, bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Steering Angle')
plt.ylabel('Count')
plt.title('Validation Dataset - Steering Angle Distribution')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Center (0°)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################
    

net = Net(num_outputs=1).to(device)  # 1 output for regression (continuous angle)


#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

# For regression tasks - Mean Squared Error Loss
# criterion = nn.MSELoss()
# You could also use L1Loss (MAE) for more robustness to outliers:
criterion = nn.L1Loss()

# Optimizer - Adam often works better for regression
optimizer = optim.Adam(net.parameters(), lr=args.lr)
# Or SGD:
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)


#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################
losses = {'train': [], 'val': []}
maes = {'train': [], 'val': []}  # Mean Absolute Error for regression
best_mae = float('inf')  # Lower is better for MAE
for epoch in range(args.epochs):  # loop over the dataset multiple times

    epoch_loss = 0.0
    epoch_mae = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, angles]
        inputs, angles = data[0].to(device), data[1].to(device)
        
        # Ensure angles are float and have correct shape
        angles = angles.float().view(-1, 1)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()

        # Calculate MAE for this batch
        mae = torch.mean(torch.abs(outputs - angles))
        
        # print statistics
        epoch_loss += loss.item()
        epoch_mae += mae.item()

    avg_loss = epoch_loss / len(trainloader)
    avg_mae = epoch_mae / len(trainloader)
    print(f'Epoch {epoch + 1} - Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}')
    losses['train'].append(avg_loss)
    maes['train'].append(avg_mae)
 
    # Validation phase - no gradients needed
    val_loss = 0
    val_mae = 0
    with torch.no_grad():
        for data in valloader:
            images, angles = data[0].to(device), data[1].to(device)
            angles = angles.float().view(-1, 1)
            
            outputs = net(images)
            loss = criterion(outputs, angles)
            mae = torch.mean(torch.abs(outputs - angles))

            val_loss += loss.item()
            val_mae += mae.item()

    avg_val_loss = val_loss / len(valloader)
    avg_val_mae = val_mae / len(valloader)
    
    losses['val'].append(avg_val_loss)
    maes['val'].append(avg_val_mae)
    
    print(f'         Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}')

    # Save best model based on validation MAE (lower is better)
    if avg_val_mae < best_mae:
        torch.save(net.state_dict(), args.model)
        best_mae = avg_val_mae
        print(f'  → New best model saved: {args.model} (val MAE: {best_mae:.4f})')

print('Finished Training')

plt.plot(losses['train'], label = 'Training')
plt.plot(losses['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(maes['train'], label = 'Training MAE')
plt.plot(maes['val'], label = 'Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Mean Absolute Error over Epochs')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
print('\n' + '='*60)
print('FINAL EVALUATION ON VALIDATION SET')
print('='*60)

net.load_state_dict(torch.load(args.model))

# Evaluation metrics
actual_angles = []
predicted_angles = []
errors = []

with torch.no_grad():
    for data in valloader:
        images, angles = data[0].to(device), data[1].to(device)
        angles = angles.float().view(-1, 1)
        
        outputs = net(images)
        
        actual_angles.extend(angles.cpu().numpy().flatten())
        predicted_angles.extend(outputs.cpu().numpy().flatten())
        errors.extend((outputs - angles).cpu().numpy().flatten())

actual_angles = np.array(actual_angles)
predicted_angles = np.array(predicted_angles)
errors = np.array(errors)

# Calculate metrics
mae = np.mean(np.abs(errors))
mse = np.mean(errors**2)
rmse = np.sqrt(mse)
r2 = 1 - (np.sum(errors**2) / np.sum((actual_angles - np.mean(actual_angles))**2))

print(f'\nRegression Metrics:')
print(f'  Mean Absolute Error (MAE):  {mae:.4f}')
print(f'  Mean Squared Error (MSE):   {mse:.4f}')
print(f'  Root Mean Squared Error:    {rmse:.4f}')
print(f'  R² Score:                   {r2:.4f}')

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(actual_angles, predicted_angles, alpha=0.5)
plt.plot([-0.5, 0.5], [-0.5, 0.5], 'r--', label='Perfect prediction')
plt.xlabel('Actual Angle')
plt.ylabel('Predicted Angle')
plt.title(f'Actual vs Predicted (R²={r2:.3f})')
plt.legend()
plt.grid(alpha=0.3)

# Error distribution
plt.subplot(1, 2, 2)
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (MAE={mae:.4f})')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Analyze errors by angle range
print(f'\nError Analysis by Angle Range:')
for angle_min, angle_max, label in [(-0.5, -0.3, 'Hard Left'), (-0.3, -0.1, 'Left'), 
                                      (-0.1, 0.1, 'Straight'), (0.1, 0.3, 'Right'), 
                                      (0.3, 0.5, 'Hard Right')]:
    mask = (actual_angles >= angle_min) & (actual_angles < angle_max)
    if np.sum(mask) > 0:
        range_mae = np.mean(np.abs(errors[mask]))
        print(f'  {label:12s} ({angle_min:+.1f} to {angle_max:+.1f}): MAE = {range_mae:.4f} ({np.sum(mask)} samples)')
