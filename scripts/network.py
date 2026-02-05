from torch import nn

import torch

class Net(nn.Module):
    def __init__(self, num_outputs=5):
        super().__init__()
        # Conv 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 6 channels divided into 3 groups (2 channels per group)
        self.gn1 = nn.GroupNorm(3, 6) 
        
        # Conv 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16 channels divided into 4 groups (4 channels per group)
        self.gn2 = nn.GroupNorm(4, 16)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv -> GroupNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.gn1(self.conv1(x))))
        x = self.pool(self.relu(self.gn2(self.conv2(x))))
        
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x





# class Net(nn.Module):
#     def __init__(self, num_outputs=5):
#         """
#         Neural network for steering prediction.
        
#         Args:
#             num_outputs: Number of output neurons
#                          5 for classification (default)
#                          1 for regression
#         """
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(1344, 256)
#         self.fc2 = nn.Linear(256, num_outputs)  # Configurable output size

#         self.relu = nn.ReLU()


#     def forward(self, x):
#         #extract features with convolutional layers
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
        
#         #linear layer for prediction
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
       
#         return x
    
    



# WORKING MODEL 
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(1344, 256)
#         self.fc2 = nn.Linear(256, 5)

#         self.relu = nn.ReLU()


#     def forward(self, x):
#         #extract features with convolutional layers
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
        
#         #linear layer for classification
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
        
#         return x





    