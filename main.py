import autoencoders
import os
import json
import torch
from torchvision import datasets, transforms
from custom_dataset import CustomDataset
from trainer import AutoencoderTrainer, AutoencoderTrainingManager, ConvClassifierTrainer, Trainer 
from autoencoder_visualizer import AutoencoderVisualizer
from autoencoders import Autoencoder_S2F, Autoencoder_F2S, Classifier

# Load configurations from a JSON file
config_file_path = './configurations.json'
with open(config_file_path, 'r') as config_file:
    configurations = json.load(config_file)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

# Se cargan los datasets
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transforms.ToTensor())
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transforms.ToTensor())

# Se cargan los autoencoders
train_set_autoencoder = CustomDataset(train_set_orig, classify=False)
valid_set_autoencoder = CustomDataset(valid_set_orig, classify=False)
trainer = AutoencoderTrainingManager(configurations, train_set_autoencoder, valid_set_autoencoder)
# trainer.train_all()

# map_labels = {
#     0: 'T-shirt/top',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
# }

# # Se cargan los clasificadores
train_set_classifier = CustomDataset(train_set_orig, classify=True)
valid_set_classifier = CustomDataset(valid_set_orig, classify=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.CrossEntropyLoss()


model_classes=[
    Autoencoder_S2F, 
    Autoencoder_S2F, 
    Autoencoder_F2S,
    Autoencoder_F2S
    ]
model_paths=[
    './results/S2F_0.001.pt',
    './results/S2F_0.001.pt',
    './results/F2S_0.001.pt',
    './results/F2S_0.001.pt'
    ]
models = []
for model_class, model_path in zip(model_classes, model_paths):
            model = model_class()
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)

# for i, (model) in enumerate(models):
#     classifier_model = Classifier(model)
#     classifier_trainer = ConvClassifierTrainer(
#             model               = classifier_model,
#             config              = configurations[i],
#             t_dataset           = train_set_classifier,
#             v_dataset           = valid_set_classifier,
#             device              = device
#             )
#     train_loss_incorrect, train_loss, valid_loss, train_accuracy_incorrect, train_accuracy, valid_accuracy = classifier_trainer.train()
#     classifier_trainer.save_results(
#            train_loss_incorrect, 
#            train_loss, valid_loss, 
#            train_accuracy_incorrect, 
#            train_accuracy, 
#            valid_accuracy
#            )




# visualizer = AutoencoderVisualizer(
#     model_classes=[Autoencoder_S2F, Autoencoder_S2F, Autoencoder_F2S, Autoencoder_F2S],
#     model_paths=[
#         './results/S2F_0.001.pt',
#         './results/S2F_0.01.pt',
#         './results/F2S_0.001.pt',
#         './results/F2S_0.01.pt',
#     ],
#     dataset=valid_set_autoencoder
# )

# # Generate a single visualization with input and outputs from all models
# visualizer.visualize_comparisons(rows=4, output_path="multi_reconstruction.png")

