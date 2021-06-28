import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tfrecord.torch.dataset import TFRecordDataset
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import time

batch_size = 64
image_size = [100, 100]

def read_tfrecord(example):
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    
    label = []
    
    for val in list(df.columns)[2:]: label.append(example[val])

    return image, label

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord)
    
    return dataset

def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    return dataset

df = pd.read_csv('preprocessed_data.csv')

tfrlist = ['data/' + x for x in os.listdir('data')]
file_names = tf.io.gfile.glob(tfrlist)

all = list(range(len(file_names)))
train_index = random.sample(all, int(len(all) * 0.7))
test_and_validation_index = list(set(all) - set(train_index))
valid_index = random.sample(test_and_validation_index, int(len(test_and_validation_index) * 0.5))
text_index = list(set(test_and_validation_index) - set(valid_index))

train_file_names, valid_file_names, test_file_names = [file_names[index] for index in train_index], [file_names[index] for index in valid_index], [file_names[index] for index in text_index]

feature_description = {}
for elem in list(df.columns)[2:]:
    feature_description[elem] = tf.io.FixedLenFeature([], tf.int64)
feature_description['image'] = tf.io.FixedLenFeature([], tf.string)

train_dataset = get_dataset(train_file_names)
valid_dataset = get_dataset(valid_file_names)
test_dataset = get_dataset(test_file_names)

# print("Train TFRecord Files:", len(train_file_names))
# print("Validation TFRecord Files:", len(valid_file_names))
# print("Test TFRecord Files:", len(test_file_names))

print(type(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

data = next(iter(train_loader))
print(data)







# class CNN(nn.Module):
#     def __init__(self, kernel_size = 5):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 5, kernel_size)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(5, 10, kernel_size)
#         self.conv_to_fc = 10 * pow(((224-kernel_size+1)//2-kernel_size+1)//2,2)
#         self.fc1 = nn.Linear(self.conv_to_fc, 32)
#         self.fc2 = nn.Linear(32, 14)    

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, self.conv_to_fc)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# def get_accuracy(model, data_loader):
#     correct = 0
#     total = 0
#     for imgs, labels in data_loader:
        
#         output = model(imgs)
        
#         #select index with maximum prediction score
#         pred = output.max(1, keepdim=True)[1]
#         correct += pred.eq(labels.view_as(pred)).sum().item()
#         total += imgs.shape[0]
#     return correct / total

# net = CNN()
# if torch.cuda.is_available():
#     net.cuda()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(net.parameters(), lr=0.01)

# train_accuracy = np.zeros(1)
# validation_accuracy = np.zeros(1)

# start_time = time.time()
# for epoch in range(1):
#     total_train_loss = 0.0
#     total_train_err = 0.0
#     total_epoch = 0
#     for i, data in enumerate(train_loader):
#         images, labels = data
#         if torch.cuda.is_available():
#             images = images.cuda()
#             labels = labels.cuda()
#         optimizer.zero_grad()
#         output = net(images)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#     train_accuracy[epoch] = get_accuracy(net, train_loader)
#     validation_accuracy[epoch] = get_accuracy(net, valid_loader)
