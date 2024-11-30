from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset, classify=False):
        self.dataset = dataset        
        self.classify = classify

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id_dataset):
        image, label = self.dataset[id_dataset]
        if self.classify:
            return image, label
        else:
            return image, image