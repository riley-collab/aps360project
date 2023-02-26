import torch
import numpy as np

def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    for _, data in enumerate(data_loader):
        images, labels = data
        output = model(images)
        correct += torch.round(torch.sigmoid(output)).eq(labels).sum().item()
        total += labels.nelement()

    return correct / total

def get_loss(model, data_loader, criterion):
    total_loss = 0
    for i, data in enumerate(data_loader):
        images, labels = data
        output = model(images)
        loss = criterion(output, labels.type_as(output))
        total_loss += loss.item()
    return total_loss / (i+1)

def train(net, train_loader, valid_loader, criterion, optimizer, num_epochs):
    train_accuracy = np.zeros(num_epochs)
    validation_accuracy = np.zeros(num_epochs)
    epochs = range(num_epochs)

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels.type_as(output))
            loss.backward()
            optimizer.step()
        train_accuracy[epoch] = get_accuracy(net, train_loader)
        validation_accuracy[epoch] = get_accuracy(net, valid_loader)
        print(f"Epoch: {epoch+1}\nTraining Loss: {get_loss(net, train_loader, criterion)}, Validation Loss: {get_loss(net, valid_loader, criterion)}, Training accuracy: {train_accuracy[epoch]}, Validation accuracy: {validation_accuracy[epoch]}")

    print("Training complete.")

    return train_accuracy, validation_accuracy, epochs
