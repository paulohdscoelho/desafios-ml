import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

def train(model, train_loader):
    #definindo a função de perda e o otimizador

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #lr = learning rate

    #treinando a rede

    for epoch in range(10): #loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            #get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.3f' % (epoch + 1, i+1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    
    PATH = './trained_cifar_net_fine_tuned.pth'
    torch.save(model.state_dict(), PATH)

