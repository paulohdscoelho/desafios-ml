from dataset import ImageDataset

from torchvision import transforms
import torch

from model import ResNet_pt


import numpy as np
from sklearn.model_selection import KFold

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
    return train_loss,train_correct

  
def valid_epoch(model, dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images,labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss=loss_fn(output,labels)
            valid_loss+=loss.item()*images.size(0)
            _, predictions = torch.max(output.data,1)
            val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct


def main():
    torch.manual_seed(42)

    num_epochs = 10
    batch_size = 8
    k = 5    

    kf = KFold(n_splits=k)

    foldperf = {}

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.46], std=[0.25])
    ])

    root_dir = "../MLChallenge_Dataset/Data"
    dataset = ImageDataset(root_dir,transform=transformation)   

    model = ResNet_pt().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #lr = learning rate   

    for fold, (train_idx,val_idx) in enumerate(kf.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
               
        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model, train_loader, criterion, optimizer)
            test_loss, test_correct=valid_epoch(model, test_loader, criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)   
        
        foldperf['fold{}'.format(fold+1)] = history  
    
    torch.save(model,'k_cross_ResNet.pt')      
    
    pickle.dump(foldperf, open('foldperf.pkl', 'wb'))

if __name__ == '__main__':
    main()    
