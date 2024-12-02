from autoencoders import Autoencoder_F2S, Autoencoder_S2F, Classifier
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
import os

model_classes=[
    Classifier(Autoencoder_S2F()), 
    Classifier(Autoencoder_F2S()), 
    ]
model_paths=[
    './results_class/ClassifierS2F_lineal.pt',
    './results_class/ClassifierF2S_lineal.pt',
    ]
models = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_class, model_path in zip(model_classes, model_paths):
            model = model_class
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

# Se cargan los datasets
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transforms.ToTensor())
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transforms.ToTensor())

valid_loader = DataLoader(valid_set_orig, batch_size = 100, shuffle = True, num_workers=os.cpu_count()-1)

# run the models on the validation set and plot the confusion matrix
for i, model in enumerate(models):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_set_orig.classes)
    fig, ax = plt.subplots(figsize=(12, 13))  # Increase the figure size
    disp.plot(ax=ax, colorbar=False)
    plt.savefig(f'./results_class/confusion_matrix_{("S2F" if not i else "F2S")}.png')
    plt.close()