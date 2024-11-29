from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id_dataset):
        image, label = self.dataset[id_dataset]
        return image, image