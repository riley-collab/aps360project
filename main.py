import preprocessing as p
import training as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size = 64
num_epochs = 25
learning_rate = 0.001
image_size = [100, 100]

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.layer1 = nn.Linear(100*100*3, 300)
        self.layer2 = nn.Linear(300, 64)
        self.layer3 = nn.Linear(64, 4)
    def forward(self, img):
        flattened = img.reshape(-1, 100*100*3)
        activation1 = self.layer1(flattened)
        activation1 = F.leaky_relu(activation1)
        activation2 = self.layer2(activation1)
        activation2 = F.leaky_relu(activation2)
        activation3 = self.layer3(activation2)
        return activation3
    
class SimpleCNN(nn.Module):
    def __init__(self, kernel_size = 5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size)
        self.conv3 = nn.Conv2d(3, 3, kernel_size)
        self.conv_to_fc = 243
        self.fc1 = nn.Linear(self.conv_to_fc, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)    

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(batch_size, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, kernel_size = 5):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_to_fc = 20
        self.fc1 = nn.Linear(self.conv_to_fc, 11)
        self.fc2 = nn.Linear(11, 4)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        for i in range(50):
            x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(batch_size, self.conv_to_fc)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_image_size():
    return image_size

def main():
    train_dataset, valid_dataset, test_dataset = p.get_datasets(sample=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    net = SimpleCNN()
    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

    train_accuracy, validation_accuracy, epochs = t.train(net, train_loader, valid_loader, criterion, optimizer, num_epochs)

    plt.plot(epochs, train_accuracy)
    plt.title("Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.savefig("training.png")

    plt.clf()

    plt.plot(epochs, validation_accuracy)
    plt.title("Validation Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.savefig("validation.png")

    print(f"Test accuracy: {t.get_accuracy(net, test_loader)}")

if __name__ == "__main__":
    main()
