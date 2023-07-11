from dataset import ImageDataset

from torchvision import transforms
import torch

from train import train
from test import test
from model import ResNet_pt

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

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #carrega o dataset de treino e teste

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    #verifica o tamanho do dataset de treino e teste
    print("Train dataset size: ", len(train_dataset))
    print("Test dataset size: ", len(test_dataset))

    model = ResNet_pt()

    train(model, train_loader)

if __name__ == '__main__':
    main()