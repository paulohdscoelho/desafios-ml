from torch.utils.data import Dataset
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self._root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._parse_dataset()

    def _parse_dataset(self):
        image_paths = []
        labels = []

        #Feature Preprocessing: Carregando os dados e  lidando com dados faltantes

        for id_folder in os.listdir(self._root_dir):
            id_folder_path = os.path.join(self._root_dir, id_folder)
            if os.path.isdir(id_folder_path):
                for class_folder in os.listdir(id_folder_path):
                    class_folder_path = os.path.join(id_folder_path, class_folder)
                    if os.path.isdir(class_folder_path): # Verificando se a pasta está vazia
                        class_label = 1 if class_folder == 'live' else 0  # Label é 1 se a pasta for live, 0 se for spoof
                        for image_file in os.listdir(class_folder_path):
                            image_path = os.path.join(class_folder_path, image_file)
                            image_paths.append(image_path)
                            labels.append(class_label)

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label