import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from steerDS import SteerDataSet

#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################
torch.manual_seed(0)

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

#transformations for raw images before going to CNN
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

script_path = os.path.dirname(os.path.realpath(__file__))

###################
## Train dataset ##
###################

# splitting randomly into training and validation

dataset = SteerDataSet(os.path.join(script_path, '..', 'dataC'), '.jpg', transform)

n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val   = n_total - n_train

train_ds, val_ds = random_split(dataset, [n_train, n_val],generator=torch.Generator().manual_seed(42))

train_ds.class_labels = dataset.class_labels
val_ds.class_labels = dataset.class_labels

#train_ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_starter'), '.jpg', transform)
print("The train dataset contains %d images " % len(train_ds))


# extract labels from training subset
train_labels = [train_ds[i][1] for i in range(len(train_ds))]

# count samples per class
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts

# weight per sample
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

trainloader = DataLoader(
    train_ds,
    batch_size=8,
    sampler=sampler
)

# validation loader (no sampler)
valloader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False
)



#data loader nicely batches images for the training process and shuffles (if desired)
#trainloader = DataLoader(train_ds,batch_size=8,shuffle=True)
all_y = []
for S in trainloader:
    im, y = S    
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Training Dataset')
plt.show()

# visualise some images and print labels -- check these seem reasonable
example_ims, example_lbls = next(iter(trainloader))
print(' '.join(f'{example_lbls[j]}' for j in range(len(example_lbls))))
imshow(torchvision.utils.make_grid(example_ims))


########################
## Validation dataset ##
########################

#val_ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'val_starter'), '.jpg', transform)
print("The train dataset contains %d images " % len(val_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
#valloader = DataLoader(val_ds,batch_size=1)
all_y = []
for S in valloader:
    im, y = S    
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Validation Dataset')
plt.show()

#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # no hardcoded size
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)
    

net = Net()

#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

#for classification tasks
#criterion = nn.CrossEntropyLoss()
criterion = nn.SmoothL1Loss()
#You could use also ADAM
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################
losses = {'train': [], 'val': []}
mae_hist = {'train': [], 'val': []}

best_val_mae = float("inf")

for epoch in range(20):

    # ---------------- TRAIN ----------------
    epoch_loss = 0.0
    epoch_abs_err = 0.0
    n_train = 0

    for inputs, labels in trainloader:
        labels = labels.float()

        optimizer.zero_grad()

        outputs = net(inputs).float()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * labels.size(0)
        epoch_abs_err += torch.sum(torch.abs(outputs - labels)).item()
        n_train += labels.size(0)

    train_loss = epoch_loss / n_train
    train_mae  = epoch_abs_err / n_train

    losses['train'].append(train_loss)
    mae_hist['train'].append(train_mae)

    # ---------------- VALIDATION ----------------
    val_loss_sum = 0.0
    val_abs_err = 0.0
    n_val = 0

    with torch.no_grad():
        for images, labels in valloader:
            labels = labels.float()
            outputs = net(images).float()

            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_abs_err += torch.sum(torch.abs(outputs - labels)).item()
            n_val += labels.size(0)

    val_loss = val_loss_sum / n_val
    val_mae  = val_abs_err / n_val

    losses['val'].append(val_loss)
    mae_hist['val'].append(val_mae)

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss:.4f}, train_MAE={train_mae:.4f}, "
          f"val_loss={val_loss:.4f}, val_MAE={val_mae:.4f}")

    if val_mae < best_val_mae:
        torch.save(net.state_dict(), 'steer_net.pth')
        best_val_mae = val_mae

print('Finished Training')

plt.plot(losses['train'], label='Training Loss')
plt.plot(losses['val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(mae_hist['train'], label='Training MAE')
plt.plot(mae_hist['val'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()


#######################################################################
# PERFORMANCE EVALUATION (REGRESSION)
#######################################################################
net.load_state_dict(torch.load('steer_net.pth'))

abs_err = 0.0
n = 0

with torch.no_grad():
    for images, labels in valloader:
        labels = labels.float()
        outputs = net(images).float()

        abs_err += torch.sum(torch.abs(outputs - labels)).item()
        n += labels.size(0)

print(f"Validation MAE: {abs_err / n:.4f}")