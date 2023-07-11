import numpy as np
import matplotlib.pyplot as plt

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