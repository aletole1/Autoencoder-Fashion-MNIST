import torch
from torch import nn

class Autoencoder_S2F(nn.Module):
    def __init__(self, dropout=0.2):
        super(Autoencoder_S2F, self).__init__()
        self.encoder = nn.Sequential(
            # Capa convolucional 1
            nn.Conv2d(1, 16, kernel_size=3, padding=0, stride=2),  # Conv2D de entrada, (1, 28, 28) a (16, 26, 26)
            nn.ReLU(),
            nn.Dropout(dropout),
            # Capa convolucional 2
            nn.Conv2d(16, 8, kernel_size=3, padding=0, stride=1),  # Conv2D de entrada, (16, 13, 13) a (16, 11, 11)
            nn.ReLU(),
            nn.Dropout(dropout),
            # # Capa lineal
            nn.Flatten(),  # Aplanamos de (8, 11, 11) a (8 * 11 * 11)
            nn.Linear(8 * 11 * 11, 8 * 11 * 11),  # Proyección lineal
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            # Capa lineal
            nn.Linear(8 * 11 * 11, 16 * 13 * 13),  # Proyección lineal
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (16, 13, 13)),  # Reconstruimos las dimensiones originales pasamos de 16*13*13 a (16, 13, 13)
            # Capa deconvolucional 1
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0, output_padding=0),  # Ajuste para 26x26, de (16, 13, 13) a (16, 27, 27)
            nn.ReLU(),
            nn.Dropout(dropout),
            # Capa deconvolucional 2
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=1, padding=0, output_padding=0),  # Ajuste para 28x28, de (16, 27, 27) a (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.lineal(x)
        x = self.decoder(x)
        return x

class Autoencoder_F2S(nn.Module):
    def __init__(self, dropout=0.2):
        super(Autoencoder_F2S, self).__init__()
        self.encoder = nn.Sequential(
            # Capa convolucional 2
            nn.Conv2d(1, 16, kernel_size=3, padding=0, stride=1),  # Conv2D de entrada, (1, 28, 28) a (16, 26, 26)
            nn.ReLU(),
            nn.Dropout(dropout),
            # Capa convolucional 1
            nn.Conv2d(16, 8, kernel_size=3, padding=0, stride=2),  # Conv2D de entrada, (16, 26, 26) a (16, 12, 12)
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # # Capa lineal
            nn.Flatten(),  # Aplanamos de (8, 12, 12) a (8 * 12 * 12)
            nn.Linear(8 * 12 * 12, 8 * 12 * 12),  # Proyección lineal
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            # Capa lineal
            nn.Linear(8 * 12 * 12, 16 * 13 * 13),  # Proyección lineal
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (16, 13, 13)),  # Reconstruimos las dimensiones originales pasamos de 16*13*13 a (16, 13, 13)
            # Capa deconvolucional 1
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0, output_padding=0),  # Ajuste para 26x26, de (16, 13, 13) a (16, 27, 27)
            nn.ReLU(),
            nn.Dropout(dropout),
            # Capa deconvolucional 2
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=1, padding=0, output_padding=0),  # Ajuste para 28x28, de (16, 27, 27) a (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Prueba de modelo
# model = Autoencoder2()
# dummy_input = torch.randn(1, 1, 28, 28)  # Batch size = 1, Canal = 1, Dimensión = 28x28
# output = model(dummy_input)
# print(f"Output shape: {output.shape}")  # Espero que salga: [1, 1, 28, 28]