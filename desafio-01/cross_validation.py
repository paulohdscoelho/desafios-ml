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
from torch.utils.data import DataLoader,SubsetRandomSampler,WeightedRandomSampler

from torchvision import transforms

from train import train_epoch
from valid import valid_epoch


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    batch_size = 8
    k = 5    

    kf = KFold(n_splits=k)

    foldperf = {}

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[], 'train_precision':[], 'test_precision':[],'train_recall' :[], 'test_recall':[], 'train_f1':[], 'test_f1':[]}

    transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.46], std=[0.25])
    ])

    root_dir = "../MLChallenge_Dataset/Data"
    dataset = ImageDataset(root_dir,transform=transformation)   

    model = ResNet_pt().to(device)

    criterion = nn.CrossEntropyLoss()
    
    #criando optimizer com learning rate de 0.001, momentum de 0.9 e weight decay de 0.001 para evitar overfitting

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001) 
    class_weights = pickle.load(open('input/weights.pkl', 'rb'))

    for fold, (train_idx,val_idx) in enumerate(kf.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))
        
        #Realizando o undersampling para balancear o dataset

        train_sampler = WeightedRandomSampler(class_weights, len(train_idx), replacement=True)
        test_sampler = WeightedRandomSampler(class_weights, len(val_idx), replacement=True)

        #Treinando sem o undersampling

        # train_sampler = SubsetRandomSampler(train_idx)
        # test_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
               
        for epoch in range(num_epochs):
            train_loss, train_correct, train_precision, train_recall, train_f1_score = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_correct, test_precision, test_recall, test_f1_score =valid_epoch(model, test_loader, criterion, device)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100
            
            train_precision = train_precision * 100
            test_precision = test_precision * 100
            train_recall = train_recall * 100
            test_recall = test_recall * 100
            train_f1_score = train_f1_score * 100
            test_f1_score = test_f1_score * 100
            
            
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} % Training Precision {:.2f} Test Precision {:.2f} Training Recall {:.2f} Test Recall {:.2f} Training f1 score {:.2f} Test f1 score {:.2f}".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc,
                                                                                                                train_precision,
                                                                                                                test_precision,
                                                                                                                train_recall,
                                                                                                                test_recall,
                                                                                                                train_f1_score,
                                                                                                                test_f1_score))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)   
            history['train_precision'].append(train_precision)
            history['test_precision'].append(test_precision)
            history['train_recall'].append(train_recall)
            history['test_recall'].append(test_recall)
            history['train_f1'].append(train_f1_score)
            history['test_f1'].append(test_f1_score)
        
        foldperf['fold{}'.format(fold+1)] = history  
    
    torch.save(model,'k_cross_ResNet_balanced.pt')      
    
    pickle.dump(foldperf, open('foldperf_balanced.pkl', 'wb'))

if __name__ == '__main__':
    main()    
