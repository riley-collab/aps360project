#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import pandas as pd
import torch
import random
import os

def read_tfrecord(example):
    df = pd.read_csv('preprocessed_data.csv')
    feature_description = {}
    for elem in list(df.columns)[1:]:
        feature_description[elem] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image'] = tf.io.FixedLenFeature([], tf.string)

    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, get_image_size())
    image = tf.cast(image, tf.float32) / 255.0
    
    label = []
    for val in list(df.columns)[1:]: 
        if val == "No Finding" or val == "Atelectasis" or val == "Infiltration" or val == "Effusion":
            label.append(example[val])

    return image, label

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord)
    
    return dataset

def tensorflow_to_pytorch(dataset):
    set = []
    for _, data in enumerate(dataset):
        images, labels = data
        images = torch.from_numpy(images.numpy()).permute(2, 0, 1)
        labels = torch.from_numpy(labels.numpy())
        if labels.sum() != 0 and (not (labels.sum() == 1 and labels[0] == 1) or random.random() < 0.33): # filter data
            set.append([images, labels])
    return set

def get_datasets(sample=False):
    tfrlist = ['data/' + x for x in os.listdir('data')]
    file_names = tf.io.gfile.glob(tfrlist)

    all = list(range(len(file_names)))
    if sample:
        all = list(range(len(file_names))[:10])
    train_index = random.sample(all, int(len(all) * 0.7))
    test_and_validation_index = list(set(all) - set(train_index))
    valid_index = random.sample(test_and_validation_index, int(len(test_and_validation_index) * 0.5))
    text_index = list(set(test_and_validation_index) - set(valid_index))

    train_file_names, valid_file_names, test_file_names = [file_names[index] for index in train_index], [file_names[index] for index in valid_index], [file_names[index] for index in text_index]

    print("Converting training data.")
    train_dataset = tensorflow_to_pytorch(load_dataset(train_file_names))
    print("Training data converted.")

    print("Converting validation data.")
    valid_dataset = tensorflow_to_pytorch(load_dataset(valid_file_names))
    print("Validation data converted.")

    print("Converting test data.")
    test_dataset = tensorflow_to_pytorch(load_dataset(test_file_names))
    print("Test data converted.")

    return train_dataset, valid_dataset, test_dataset


# In[3]:


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
            if i % 100 == 0:
                print(f"Epoch: {epoch+1} Iteration: {i}\nTraining Loss: {get_loss(net, train_loader, criterion)}, Validation Loss: {get_loss(net, valid_loader, criterion)}, Training accuracy: {get_accuracy(net, train_loader)}, Validation accuracy: {get_accuracy(net, valid_loader)}")
        train_accuracy[epoch] = get_accuracy(net, train_loader)
        validation_accuracy[epoch] = get_accuracy(net, valid_loader)
        print(f"Epoch: {epoch+1}\nTraining Loss: {get_loss(net, train_loader, criterion)}, Validation Loss: {get_loss(net, valid_loader, criterion)}, Training accuracy: {train_accuracy[epoch]}, Validation accuracy: {validation_accuracy[epoch]}")

    print("Training complete.")

    return train_accuracy, validation_accuracy, epochs


# In[4]:


def get_image_size():
    return image_size


# In[5]:



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self, kernel_size = 5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size)
        self.conv3 = nn.Conv2d(3, 3, kernel_size)
        self.conv_to_fc = 1452
        self.fc1 = nn.Linear(self.conv_to_fc, 600)
        self.fc2 = nn.Linear(600, 32)
        self.fc3 = nn.Linear(32, 4)    

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(batch_size, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x =self.fc3(x)

        return x
    


# In[6]:


batch_size = 64
num_epochs = 10
learning_rate = 0.001
image_size = [100, 100]

train_dataset, valid_dataset, test_dataset = get_datasets(sample=True)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)


# In[9]:


net = SimpleCNN()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

train_accuracy, validation_accuracy, epochs = train(net, train_loader, valid_loader, criterion, optimizer, num_epochs)

plt.plot(epochs, train_accuracy)
plt.title("Training Curve")
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.show()

plt.plot(epochs, validation_accuracy)
plt.title("Validation Curve")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.show()

print(f"Test accuracy: {get_accuracy(net, test_loader)}")


# In[19]:


def get_unseen_dataset(sample=False):
    tfrlist = ['unseen_data/' + x for x in os.listdir('unseen_data')]
    file_names = tf.io.gfile.glob(tfrlist)
    file_names.remove('unseen_data/.DS_Store')
    all = list(range(len(file_names)))

    unseen_index = all

    unseen_file_names= [file_names[index] for index in unseen_index]

    print("Converting unseen data.")
    unseen_dataset = tensorflow_to_pytorch(load_dataset(unseen_file_names))
    print("Unseen data converted.")


    return unseen_dataset


# In[20]:


unseen_dataset = get_unseen_dataset()

unseen_loader = torch.utils.data.DataLoader(unseen_dataset, shuffle=True, batch_size=64, drop_last=True)


# In[22]:


print(f"Unseen Data accuracy: {get_accuracy(net, unseen_loader)}")


# In[ ]:




