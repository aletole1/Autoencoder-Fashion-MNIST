import Autoencoder
import os
import json
import torch
from torchvision import datasets, transforms
from CustomDataset import CustomDataset
from Trainer import Trainer

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
        (modelo_1, train_loss_incorrect_1, train_loss_1, valid_loss_1),
        (modelo_2, train_loss_incorrect_2, train_loss_2, valid_loss_2),
        ...
    ]
    """
    results = {}
    for i,(config) in enumerate(configs):
        model, train_loss_incorrect, train_loss, valid_loss = train_configuration(config, t_dataset, v_dataset)
        results = {
            "config_"+str(i): config,
            # "model_"+str(i): model,
            "train_loss_incorrect_"+str(i): train_loss_incorrect,
            "train_loss_"+str(i): train_loss,
            "valid_loss_"+str(i): valid_loss
        }
        file_name = ""
        if config["S2F"]:
            file_name = f'./results/S2F_{config["learning_rate"]}.json'
        else:
            file_name = f'./results/F2S_{config["learning_rate"]}.json'

        if not os.path.exists('./results'):
            os.makedirs('./results')

        with open(file_name, 'w') as f:
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
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# Se cargan los datasets
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transform)
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)

train_set_autoencoder = CustomDataset(train_set_orig)
valid_set_autoencoder = CustomDataset(valid_set_orig)

train_all(configurations, train_set_autoencoder, valid_set_autoencoder)