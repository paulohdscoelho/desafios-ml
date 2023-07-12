import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def train_epoch(model, dataloader,loss_fn,optimizer, device):
    train_loss,train_correct=0.0,0
    true_labels,predicted_labels=[],[]
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
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.tolist())

    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    return train_loss,train_correct, precision, recall, f1

