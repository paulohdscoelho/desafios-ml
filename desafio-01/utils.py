import numpy as np
import matplotlib.pyplot as plt

def mostra_sumario(foldperf):

    test_loss_f,train_loss_f,test_acc_f,train_acc_f=[],[],[],[]
    train_precision_f,train_recall_f,train_f1_f=[],[],[]
    test_precision_f,test_recall_f,test_f1_f=[],[],[]

    k=5
    for folder in range(1,k+1):

         train_loss_f.append(np.mean(foldperf['fold{}'.format(folder)]['train_loss']))
         test_loss_f.append(np.mean(foldperf['fold{}'.format(folder)]['test_loss']))

         train_acc_f.append(np.mean(foldperf['fold{}'.format(folder)]['train_acc']))
         test_acc_f.append(np.mean(foldperf['fold{}'.format(folder)]['test_acc']))

         train_precision_f.append(np.mean(foldperf['fold{}'.format(folder)]['train_precision']))
         test_precision_f.append(np.mean(foldperf['fold{}'.format(folder)]['test_precision']))

         train_recall_f.append(np.mean(foldperf['fold{}'.format(folder)]['train_recall']))
         test_recall_f.append(np.mean(foldperf['fold{}'.format(folder)]['test_recall']))

         train_f1_f.append(np.mean(foldperf['fold{}'.format(folder)]['train_f1']))
         test_f1_f.append(np.mean(foldperf['fold{}'.format(folder)]['test_f1']))
         
         

    print('Performance do {} fold cross validation'.format(k))     

    print(f"Average Training Loss: {np.mean(train_loss_f):.3f} \t Average Test Loss: {np.mean(test_loss_f):.3f} \t Average Training Acc: {np.mean(train_acc_f):.2f} \t Average Test Acc: {np.mean(test_acc_f):.2f}")
    print(f"Training Precision: {np.mean(train_precision_f):.3f} \t Test Precision: {np.mean(test_precision_f):.3f} \t Training Recall: {np.mean(train_recall_f):.2f} \t Test Recall: {np.mean(test_recall_f):.2f}")
    print(f"Training F1: {np.mean(train_f1_f):.3f} \t Test F1: {np.mean(test_f1_f):.3f}")

def plota_graficos(foldperf):
      k = 5

      diz_ep = {'train_loss_ep':[],'test_loss_ep':[],'train_acc_ep':[],'test_acc_ep':[]}

      for i in range(10):
            diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
            diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_loss'][i] for f in range(k)]))
            diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(k)]))
            diz_ep['test_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_acc'][i] for f in range(k)]))

      # Create a figure with two subplots
      fig, axs = plt.subplots(1, 2, figsize=(16, 8))

      # Plot losses
      axs[0].semilogy(diz_ep['train_loss_ep'], label='Train')
      axs[0].semilogy(diz_ep['test_loss_ep'], label='Test')
      axs[0].set_xlabel('Epoch')
      axs[0].set_ylabel('Loss')
      axs[0].legend()
      axs[0].set_title('Resnet balanced loss')

      # Plot accuracies
      axs[1].semilogy(diz_ep['train_acc_ep'], label='Train')
      axs[1].semilogy(diz_ep['test_acc_ep'], label='Test')
      axs[1].set_xlabel('Epoch')
      axs[1].set_ylabel('Accuracy')
      axs[1].legend()
      axs[1].set_title('Resnet balanced accuracy')

      # Adjust the spacing between subplots
      plt.tight_layout()

      # Show the combined plot
      plt.show()
    


def class_imbalance(dataset):
    class_counts = np.bincount(dataset.labels)
    num_classes = len(class_counts)
    class_indices = np.arange(num_classes)

    class_ratios = np.round(class_counts / len(dataset.labels),2)
    print(f"class_ratios: {class_ratios}")

    plt.bar(class_indices, class_counts)
    plt.xticks(class_indices, ['spoof', 'live'])
    plt.xlabel('Label da classe')
    plt.ylabel('Frequência')
    plt.xticks(class_indices)
    plt.title('Distribuição de classe')
    plt.show()

def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()