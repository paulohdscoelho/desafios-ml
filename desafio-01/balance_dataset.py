import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision import transforms

from dataset import ImageDataset

import pickle

def perform_random_oversampling(dataset, minority_class_label, oversampling_factor):
    minority_samples = []
    majority_samples = []

    # Separate the data by class
    for sample, label in dataset:
        if label == minority_class_label:
            minority_samples.append(sample)
        else:
            majority_samples.append(sample)

    # Calculate the number of additional samples to generate
    num_minority_samples = len(minority_samples)
    num_majority_samples = len(majority_samples)
    num_samples_needed = oversampling_factor * num_majority_samples - num_minority_samples

    # Perform random oversampling
    oversampled_samples = minority_samples.copy()
    while len(oversampled_samples) < num_minority_samples + num_samples_needed:
        random_index = torch.randint(0, num_minority_samples, (1,))
        oversampled_samples.append(minority_samples[random_index])

    # Combine the oversampled data with the original majority class data
    balanced_samples = majority_samples + oversampled_samples
    balanced_labels = [0] * num_majority_samples + [1] * len(oversampled_samples)

    return balanced_samples, balanced_labels



if __name__ == '__main__':
    oversampling_factor = 2
    minority_class_label = 1

    # Load the dataset

    transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.46], std=[0.25])
    ])

    root_dir = "../MLChallenge_Dataset/Data"
    dataset = ImageDataset(root_dir,transform=transformation)   

    balanced_samples, balanced_labels = perform_random_oversampling(dataset, minority_class_label, oversampling_factor)

    # Create a new balanced dataset
    balanced_dataset = [(sample, label) for sample, label in zip(balanced_samples, balanced_labels)]

    #convert from list to Dataset
    balanced_dataset = torch.utils.data.Dataset(balanced_dataset)

    # Create a data loader for the balanced dataset
    batch_size = 32
    torch.save(balanced_dataset, 'balanced_dataset.pt')

    