# project/
# │
# ├── data/
# │   └── data_processing.py    # Script for data processing
# │
# ├── models/
# │   ├── architecture.py       # Model class and instantiated object
# │   ├── train.py              # Training script
# │   └── evaluate.py           # Evaluation script
# │
# ├── utils/
# │   └── dataset.py            # Custom dataset class
# │
# └── main.py                   # Main script to run the project


#### Step 2: create a CNN model ####

import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self): 
        super(MyCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 5),
            nn.ReLU(), 
            nn.MaxPool2d(2)
        ) 
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(128*10*10, 1000)
        self.fc2 = nn.Linear(1000, 20)  # how many classes we have
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return(x)
    
model = MyCNN()
        
