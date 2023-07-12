from dataset import ImageDataset

from torchvision import transforms
import torch
import pickle

def compute_class_weights(dataset):
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    num_samples = len(dataset)
    class_weights = [num_samples / class_counts[label] for _, label in dataset]

    return class_weights

def main():

    torch.manual_seed(42)

    #utilizando um script para cálculo de média e desvio padrão eu pude calcular quais os valores do dataset, então realizei a normalização dos dados

    transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.46], std=[0.25])
    ])

    root_dir = "../MLChallenge_Dataset/Data"
    dataset = ImageDataset(root_dir,transform=transformation)

    weights = compute_class_weights(dataset)                                 
    pickle.dump(weights, open("weights.pkl", "wb"))

if __name__ == '__main__':
    main()