#!/usr/bin/env python3
"""
Fine-Tuning Script for Steering Classifier

This script loads a pre-trained model and continues training on new data.
Useful for:
- Adding new scenarios/conditions to an existing model
- Recovering from model degradation
- Incremental learning from runtime-recorded data

Usage:
    python finetune_net.py --pretrained steer_net.pth --data runtime_recorded --model steer_net_finetuned.pth
"""

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
from torch.utils.data import random_split, ConcatDataset
from steerDS import SteerDataSet
from preprocess import PreProcessImage
from network import Net
from net_utils import get_transform

#######################################################################################################################################
####     FINE-TUNING CONFIGURATION                                                                                                ####
#######################################################################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Fine-tune steering angle classifier')

# Model loading
parser.add_argument('--pretrained', type=str, required=True,
                    help='Path to pre-trained model to fine-tune (REQUIRED)')

# Data configuration
parser.add_argument('--data', type=str, required=True,
                    help='Name of the NEW training data folder in data/ (REQUIRED)')
parser.add_argument('--original-data', type=str, default=None,
                    help='Optional: original training data to mix with new data')
parser.add_argument('--mix-ratio', type=float, default=0.5,
                    help='Ratio of original data to new data (0.0-1.0, default: 0.5)')

# Output configuration  
parser.add_argument('--model', type=str, default='steer_net_finetuned.pth',
                    help='Name for the fine-tuned model file (default: steer_net_finetuned.pth)')

# Training hyperparameters
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of fine-tuning epochs (default: 10, fewer than from-scratch)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size for training (default: 8)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate (default: 0.0001, lower than from-scratch)')

# Fine-tuning strategy
parser.add_argument('--freeze-conv', action='store_true',
                    help='Freeze convolutional layers, only train FC layers')
parser.add_argument('--gradual-unfreeze', action='store_true',
                    help='Gradually unfreeze layers during training')

args = parser.parse_args()

print("=" * 70)
print("FINE-TUNING CONFIGURATION")
print("=" * 70)
print(f"Pre-trained model: {args.pretrained}")
print(f"New data folder: {args.data}") 
print(f"Original data: {args.original_data if args.original_data else 'Not using'}")
print(f"Output model: {args.model}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.lr} (lower than from-scratch)")
print(f"Strategy: {'Freeze conv layers' if args.freeze_conv else 'Train all layers'}")
if args.gradual_unfreeze:
    print(f"  + Gradual unfreezing enabled")
print("=" * 70)

# Check if pretrained model exists
if not os.path.exists(args.pretrained):
    print(f"\\n❌ ERROR: Pre-trained model '{args.pretrained}' not found!")
    print("Please provide a valid model file with --pretrained")
    exit(1)

# Check if output model file already exists
if os.path.exists(args.model):
    print(f"\\n⚠ WARNING: Output model file '{args.model}' already exists!")
    while True:
        response = input("Do you want to (o)verride it or use a (d)ifferent name? [o/d]: ").strip().lower()
        if response == 'o':
            print(f"Will override existing model '{args.model}'")
            break
        elif response == 'd':
            new_name = input(f"Enter new model name (e.g., 'steer_net_finetuned_v2.pth'): ").strip()
            if new_name:
                args.model = new_name
                print(f"Will save model as '{args.model}'")
                break
            else:
                print("Invalid name. Please try again.")
        else:
            print("Please enter 'o' for override or 'd' for different name.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\\nRUNNING ON {device}")

transform = get_transform()
script_path = os.path.dirname(os.path.realpath(__file__))

#######################################################################################################################################
####     LOAD PRE-TRAINED MODEL                                                                                                   ####
#######################################################################################################################################

print(f"\\nLoading pre-trained model from {args.pretrained}...")
net = Net().to(device)

try:
    net.load_state_dict(torch.load(args.pretrained, map_location=device))
    print("✅ Pre-trained model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading pre-trained model: {e}")
    exit(1)

# Optional: Freeze convolutional layers for transfer learning
if args.freeze_conv:
    print("\\n🔒 Freezing convolutional layers (conv1, conv2)...")
    for name, param in net.named_parameters():
        if 'conv' in name:
            param.requires_grad = False
            print(f"   Frozen: {name}")
    print("Only FC layers will be trained.")

#######################################################################################################################################
####     SETUP DATASET                                                                                                            ####
#######################################################################################################################################

print(f"\\nLoading new data from: data/{args.data}")
new_data_path = os.path.join(script_path, '..', 'data', args.data)
new_ds = SteerDataSet(new_data_path, '.jpg', transform)
print(f"  New data samples: {len(new_ds)}")

# Optionally mix with original training data
if args.original_data:
    print(f"\\nMixing with original data from: data/{args.original_data}")
    original_data_path = os.path.join(script_path, '..', 'data', args.original_data)
    original_ds = SteerDataSet(original_data_path, '.jpg', transform)
    
    # Sample from original data based on mix_ratio
    original_size = int(len(new_ds) * args.mix_ratio / (1 - args.mix_ratio))
    if original_size > len(original_ds):
        original_size = len(original_ds)
        print(f"  Warning: Not enough original data. Using all {original_size} samples.")
    else:
        print(f"  Using {original_size} samples from original data (mix ratio: {args.mix_ratio:.2f})")
    
    # Random sample from original dataset
    original_subset, _ = random_split(original_ds, [original_size, len(original_ds) - original_size])
    
    # Concatenate new and original data
    full_ds = ConcatDataset([new_ds, original_subset])
    print(f"  Total dataset size: {len(full_ds)} (new: {len(new_ds)}, original: {original_size})")
else:
    full_ds = new_ds
    print(f"  Using only new data: {len(full_ds)} samples")

# Split into train/val (80/20)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

print(f"\\nDataset split:")
print(f"  Training: {train_size} samples")
print(f"  Validation: {val_size} samples")

#data loader
trainloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_ds, batch_size=1)

# Visualize distribution
all_y = []
for S in trainloader:
    im, y = S    
    all_y += y.tolist()

print(f'\\nInput to network shape: {im.shape}')

all_lbls, all_counts = np.unique(all_y, return_counts = True)
class_counts = np.bincount(all_y)
num_classes = len(class_counts)

plt.figure(figsize=(10, 5))
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Fine-tuning Dataset Distribution')
plt.show()

#######################################################################################################################################
####     INITIALIZE OPTIMIZER AND LOSS                                                                                            ####
#######################################################################################################################################

criterion = nn.CrossEntropyLoss()

# Use lower learning rate for fine-tuning
# Adam often works better for fine-tuning than SGD
optimizer = optim.Adam(net.parameters(), lr=args.lr)
print(f"\\nOptimizer: Adam with lr={args.lr}")

#######################################################################################################################################
####     FINE-TUNING LOOP                                                                                                         ####
#######################################################################################################################################

print(f"\\nStarting fine-tuning for {args.epochs} epochs...")
print("=" * 70)

losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0

for epoch in range(args.epochs):
    
    # Gradual unfreezing: unfreeze one layer at a time
    if args.gradual_unfreeze and epoch > 0:
        # Unfreeze layers progressively
        layers_to_unfreeze = ['conv2', 'conv1']  # Reverse order
        if epoch - 1 < len(layers_to_unfreeze):
            layer_name = layers_to_unfreeze[epoch - 1]
            print(f"\\n🔓 Unfreezing layer: {layer_name}")
            for name, param in net.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    net.train()  # Set to training mode
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = epoch_loss / len(trainloader)
    train_acc = 100. * correct / total
    losses['train'].append(train_loss)
    accs['train'].append(train_acc)
    
    print(f'Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%', end='')
 
    # Validation
    net.eval()  # Set to evaluation mode
    val_loss = 0
    # Get class labels (handle both ConcatDataset and regular Dataset)
    val_class_labels = val_ds.dataset.datasets[0].class_labels if hasattr(val_ds.dataset, 'datasets') else val_ds.dataset.class_labels
    correct_pred = {classname: 0 for classname in val_class_labels}
    total_pred = {classname: 0 for classname in val_class_labels}

    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device) 
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[val_class_labels[label.item()]] += 1
                total_pred[val_class_labels[label.item()]] += 1

    class_accs = []
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_accs.append(accuracy)
        else:
            class_accs.append(0.0)

    val_acc = np.mean(class_accs)
    val_loss_avg = val_loss / len(valloader)
    losses['val'].append(val_loss_avg)
    accs['val'].append(val_acc)
    
    print(f' | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%', end='')

    if val_acc > best_acc:
        torch.save(net.state_dict(), args.model)
        best_acc = val_acc
        print(f' → ✅ New best! Saved to {args.model}')
    else:
        print()

print("=" * 70)
print(f'Fine-tuning complete!')
print(f'Best validation accuracy: {best_acc:.2f}%')
print(f'Fine-tuned model saved to: {args.model}')
print("=" * 70)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(losses['train'], label='Training', marker='o')
ax1.plot(losses['val'], label='Validation', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Fine-tuning Loss')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(accs['train'], label='Training', marker='o')
ax2.plot(accs['val'], label='Validation', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Fine-tuning Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

#######################################################################################################################################
####     FINAL EVALUATION                                                                                                         ####
#######################################################################################################################################

print(f"\\nFinal evaluation on validation set...")
net.load_state_dict(torch.load(args.model))
net.eval()

# Get class labels (handle both ConcatDataset and regular Dataset)
final_class_labels = val_ds.dataset.datasets[0].class_labels if hasattr(val_ds.dataset, 'datasets') else val_ds.dataset.class_labels
correct_pred = {classname: 0 for classname in final_class_labels}
total_pred = {classname: 0 for classname in final_class_labels}

actual = []
predicted = []

with torch.no_grad():
    for data in valloader:
        images, labels = data[0].to(device), data[1].to(device) 
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        actual += labels.tolist()
        predicted += predictions.tolist()

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[final_class_labels[label.item()]] += 1
            total_pred[final_class_labels[label.item()]] += 1

# Confusion matrix
cm = metrics.confusion_matrix(actual, predicted, normalize = 'true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_class_labels)
disp.plot()
plt.title('Fine-tuned Model - Confusion Matrix')
plt.show()

# Per-class accuracy
print(f"\\nPer-class accuracy:")
for classname, correct_count in correct_pred.items():
    if total_pred[classname] > 0:
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'  {classname:15s}: {accuracy:.1f}% ({correct_count}/{total_pred[classname]})')
    else:
        print(f'  {classname:15s}: N/A (no samples)')

print(f"\\n✅ Fine-tuning complete! Model saved to: {args.model}")
