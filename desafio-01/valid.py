import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def valid_epoch(model, dataloader,loss_fn, device):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    true_labels,predicted_labels=[],[]
    with torch.no_grad():
        for data in dataloader:
            images,labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss=loss_fn(output,labels)
            valid_loss+=loss.item()*images.size(0)
            _, predictions = torch.max(output.data,1)
            val_correct+=(predictions == labels).sum().item()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())
            

    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return valid_loss,val_correct, precision, recall, f1
