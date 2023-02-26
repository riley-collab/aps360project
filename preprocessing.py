import tensorflow as tf
import pandas as pd
import torch
import random
import os
import main as m

def read_tfrecord(example):
    df = pd.read_csv('preprocessed_data.csv')
    feature_description = {}
    for elem in list(df.columns)[1:]:
        feature_description[elem] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image'] = tf.io.FixedLenFeature([], tf.string)

    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, m.get_image_size())
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
