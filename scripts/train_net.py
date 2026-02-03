import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from steerDS import SteerDataSet

torch.manual_seed(0)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:, :, ::-1]
    plt.imshow(rgbimg)
    plt.axis("off")
    plt.show()

# --------------------------
# Crop bottom half
# --------------------------
def crop_bottom_half(img):
    w, h = img.size
    return img.crop((0, h // 2, w, h))

# --------------------------
# Transforms
# --------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_bottom_half),
    transforms.Resize((40, 60)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_bottom_half),
    transforms.Resize((40, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# --------------------------
# Datasets
# --------------------------
script_path = os.path.dirname(os.path.realpath(__file__))

train_ds = SteerDataSet(os.path.join(script_path, "..", "data", "train"), ".jpg", train_transform)
val_ds   = SteerDataSet(os.path.join(script_path, "..", "data", "val"), ".jpg", val_transform)

print(f"Train images: {len(train_ds)}")
print(f"Val images:   {len(val_ds)}")

trainloader = DataLoader(train_ds, batch_size=8, shuffle=True)
valloader   = DataLoader(val_ds, batch_size=1, shuffle=False)

# sanity check
example_ims, example_lbls = next(iter(trainloader))
print("Input shape:", example_ims.shape)
imshow(torchvision.utils.make_grid(example_ims))

# --------------------------
# Compute class weights (from train set)
# --------------------------
counts = np.zeros(len(train_ds.class_labels), dtype=np.float32)
for _, y in train_ds:
    counts[int(y)] += 1

weights = 1.0 / torch.tensor(counts, dtype=torch.float32)
weights = weights / weights.sum() * len(counts)

print("Class labels:", train_ds.class_labels)
print("Class counts:", counts)
print("Class weights:", weights)

# --------------------------
# Model (small + regularized)
# --------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 5),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 12, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

net = Net()

# --------------------------
# Loss + optimizer
# --------------------------
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)

# --------------------------
# Train
# --------------------------
losses = {"train": [], "val": []}
accs   = {"train": [], "val": []}
best_acc = -1.0

for epoch in range(20):
    # TRAIN
    net.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

    train_loss = train_loss_sum / len(trainloader)
    train_acc = 100.0 * train_correct / train_total

    # VAL (mean-per-class accuracy like you used)
    net.eval()
    val_loss_sum = 0.0
    correct_pred = {c: 0 for c in val_ds.class_labels}
    total_pred   = {c: 0 for c in val_ds.class_labels}

    with torch.no_grad():
        for images, labels in valloader:
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item()

            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                cname = val_ds.class_labels[label.item()]
                if label == pred:
                    correct_pred[cname] += 1
                total_pred[cname] += 1

    val_loss = val_loss_sum / len(valloader)

    class_accs = []
    for cname in val_ds.class_labels:
        acc = 100.0 * correct_pred[cname] / total_pred[cname] if total_pred[cname] > 0 else 0.0
        class_accs.append(acc)

    val_acc = float(np.mean(class_accs))

    losses["train"].append(train_loss)
    losses["val"].append(val_loss)
    accs["train"].append(train_acc)
    accs["val"].append(val_acc)

    print(f"Epoch {epoch+1:02d} | train loss {train_loss:.4f} acc {train_acc:.2f}% | val loss {val_loss:.4f} mean-class-acc {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), "steer_net.pth")

print("Finished. Best mean-class-acc:", best_acc)

plt.plot(losses["train"], label="Training")
plt.plot(losses["val"], label="Validation")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

plt.plot(accs["train"], label="Training")
plt.plot(accs["val"], label="Validation")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.show()

# --------------------------
# Evaluate
# --------------------------
net.load_state_dict(torch.load("steer_net.pth"))
net.eval()

actual, predicted = [], []
with torch.no_grad():
    for images, labels in valloader:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        actual += labels.tolist()
        predicted += preds.tolist()

cm = metrics.confusion_matrix(actual, predicted, normalize="true")
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_ds.class_labels)
disp.plot(); plt.show()


'''
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

############################
# Transforms
############################
def crop_bottom_half(img):
    """
    img: PIL Image
    keeps bottom 50% of the image
    """
    w, h = img.size
    return img.crop((0, h // 2, w, h))

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_bottom_half),   # 👈 CROP HERE
    transforms.Resize((40, 60)),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.RandomAffine(
        degrees=2,
        translate=(0.02, 0.02)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_bottom_half),   # 👈 SAME CROP
    transforms.Resize((40, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



script_path = os.path.dirname(os.path.realpath(__file__))

###################
## Train dataset ##
###################

#train_ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_starter'), '.jpg', transform)

train_ds = SteerDataSet(
    os.path.join(script_path, '..', 'data', 'train_starter'),
    '.jpg',
    train_transform
)

print("The train dataset contains %d images " % len(train_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
trainloader = DataLoader(train_ds,batch_size=8,shuffle=True)
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

val_ds = SteerDataSet(
    os.path.join(script_path, '..', 'data', 'val_starter'),
    '.jpg',
    val_transform
)

print("The train dataset contains %d images " % len(val_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
valloader = DataLoader(val_ds,batch_size=1)
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
       
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)

        self.relu = nn.ReLU()




class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 5), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 5), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 12, 128),  # 2688
                nn.Dropout(0.3),
                nn.Linear(128, 5)
            )

        def forward(self, x):
            return self.classifier(self.features(x))
            

net = Net()

#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

#for classification tasks
criterion = nn.CrossEntropyLoss()
#You could use also ADAM
optimizer = optim.Adam(net.parameters(), lr=0.0001)
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################
losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0

for epoch in range(20):

    # ---- TRAIN ----
    net.train()
    epoch_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # ---- VALIDATION ----
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

for epoch in range(20):  # loop over the dataset multiple times

    epoch_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

   print(f'Epoch {epoch + 1} loss: {epoch_loss / len(trainloader)}')
    losses['train'] += [epoch_loss / len(trainloader)]
    accs['train'] += [100.*correct/total]
 
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in val_ds.class_labels}
    total_pred = {classname: 0 for classname in val_ds.class_labels}

    # again no gradients needed
    val_loss = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[val_ds.class_labels[label.item()]] += 1
                total_pred[val_ds.class_labels[label.item()]] += 1

    # print accuracy for each class
    class_accs = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accs += [accuracy]

    accs['val'] += [np.mean(class_accs)]
    losses['val'] += [val_loss/len(valloader)]

    if np.mean(class_accs) > best_acc:
        torch.save(net.state_dict(), 'steer_net.pth')
        best_acc = np.mean(class_accs)

print('Finished Training')

plt.plot(losses['train'], label = 'Training')
plt.plot(losses['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(accs['train'], label = 'Training')
plt.plot(accs['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
net.load_state_dict(torch.load('steer_net.pth'))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in valloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in val_ds.class_labels}
total_pred = {classname: 0 for classname in val_ds.class_labels}

# again no gradients needed
actual = []
predicted = []
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        actual += labels.tolist()
        predicted += predictions.tolist()

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[val_ds.class_labels[label.item()]] += 1
            total_pred[val_ds.class_labels[label.item()]] += 1

cm = metrics.confusion_matrix(actual, predicted, normalize = 'true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=val_ds.class_labels)
disp.plot()
plt.show()

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')
'''