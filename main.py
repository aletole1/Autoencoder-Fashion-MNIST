import Autoencoder
import os
import json
import torch
from torchvision import datasets, transforms
from CustomDataset import CustomDataset
from Trainer import Trainer
from Visualizer import AutoencoderVisualizer
from Autoencoder import Autoencoder_S2F, Autoencoder_F2S 

def train_configuration(config, t_dataset, v_dataset):
    """ 
    Entrena y valida un autoencoder con los datos de t_dataset y v_dataset

    La función recibe un diccionario con la configuración del autoencoder de la forma:
    {
        "learning_rate": float,
        "dropout": float,
        "batch_size": int,
        "epochs": int,
        "lineal": bool True si el modelo tiene capa lienal intermedia, False c.c. 
    } 
    """
    lr = config["learning_rate"]
    dropout = config["dropout"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Se crea el autoencoder
    if config["S2F"]:
        model = Autoencoder.Autoencoder_S2F(dropout)
    else:
        model = Autoencoder.Autoencoder_F2S(dropout)

    # Se envía el modelo al dispositivo
    model.to(device)

    # Se crea el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Se crea el DataLoader para los datos de entrenamiento y validación
    train_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1)
    valid_loader = torch.utils.data.DataLoader(v_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1)

    # Se entrena el modelo
    trainer = Trainer(model, train_loader, valid_loader, loss_fn, optimizer, device, epochs)
    train_loss_incorrect, train_loss, valid_loss = trainer.train_model()

    return model, train_loss_incorrect, train_loss, valid_loss


def train_all(configs, t_dataset, v_dataset):
    """
    Entrena y valida todos los autoencoders con las configuraciones en configs

    La función recibe una lista de diccionarios con las configuraciones de los autoencoders de la forma:
    [
        {
            "learning_rate": float,
            "dropout": float,
            "batch_size": int,
            "epochs": int,
            "lineal": bool True si el modelo tiene capa lienal intermedia, False c.c. 
        },
        ...
    ] 

    Retorna una lista de tuplas con la forma:
    [
        (config, train_loss_incorrect, train_loss, valid_loss),
        (config, train_loss_incorrect, train_loss, valid_loss),
        ...
    ]
    """
    results = {}
    for i,(config) in enumerate(configs):
        model, train_loss_incorrect, train_loss, valid_loss = train_configuration(config, t_dataset, v_dataset)
        results = {
            "config": config,
            # "model": model,
            "train_loss_incorrect": train_loss_incorrect,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        }
        file_name = ""
        if config["S2F"]:
            file_name = f'./results/S2F_{config["learning_rate"]}'
        else:
            file_name = f'./results/F2S_{config["learning_rate"]}'
        
        torch.save(model.state_dict(), f'{file_name}.pt')
        if not os.path.exists('./results'):
            os.makedirs('./results')

        with open(f'{file_name}.json', 'w') as f:
            json.dump(results, f)


configurations = [
    { # configuracion default
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 100,
        "epochs": 60,
        "S2F": True
    },
    { # configuracion default
        "learning_rate": 0.01,
        "dropout": 0.2,
        "batch_size": 100,
        "epochs": 60,
        "S2F": True
    },
    {
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 100,
        "epochs": 60,
        "S2F": False
    },
    {
        "learning_rate": 0.01,
        "dropout": 0.2,
        "batch_size": 100,
        "epochs": 60,
        "S2F": False
    },
]

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

# Se cargan los datasets
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transforms.ToTensor())
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transforms.ToTensor())

train_set_autoencoder = CustomDataset(train_set_orig)
valid_set_autoencoder = CustomDataset(valid_set_orig)

# train_all(configurations, train_set_autoencoder, valid_set_autoencoder)


visualizer = AutoencoderVisualizer(
    model_classes=[Autoencoder_S2F, Autoencoder_S2F, Autoencoder_F2S, Autoencoder_F2S],
    model_paths=[
        './results/S2F_0.001.pt',
        './results/S2F_0.01.pt',
        './results/F2S_0.001.pt',
        './results/F2S_0.01.pt',
    ],
    dataset=train_set_autoencoder
)

# Generate a single visualization with input and outputs from all models
visualizer.visualize_comparisons(rows=4, output_path="multi_reconstruction.png")