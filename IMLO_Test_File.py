"""# Load and Test the trained model"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

testTransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

testData = datasets.Flowers102(
    root = "./datasets",
    split = "test",
    transform = testTransform,
    download = True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self, classAmount):
        super(NeuralNetwork, self).__init__()
        self.convStack =  nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.5)
        )

        dummyInput = torch.zeros(1, 3, 256, 256)
        dummyOutput = self.convStack(dummyInput)
        self.convOutputSize = dummyOutput.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.convOutputSize, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(512, classAmount)
        )


    def forward(self, x):
        x = self.convStack(x)
        x = self.classifier(x)
        return x

newModel = NeuralNetwork(102).to(device)

newModel.load_state_dict(torch.load('bestModel.pt', map_location=torch.device('cpu')))
newModel.eval()

def testTrainedModel(dataloader, model, lossFunction):
    model.eval()
    total = 0
    correct = 0
    testLoss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total += y.size(0)
            testLoss += lossFunction(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    testLoss = testLoss/total
    correct = (correct/total)  * 100
    print(f"Testing: Accuracy: {(correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")

testDataloader = DataLoader(testData, batch_size = 32, shuffle = False, num_workers = 6)
lossFunction = nn.CrossEntropyLoss()

if __name__ == "__main__":
    testTrainedModel(testDataloader, newModel, lossFunction)