from autoencoder_visualizer import AutoencoderVisualizer
from autoencoders import Autoencoder_S2F, Autoencoder_F2S
from datasets import train_set_autoencoder

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