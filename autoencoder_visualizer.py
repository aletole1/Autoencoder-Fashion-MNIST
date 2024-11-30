import torch
import matplotlib.pyplot as plt
import os

class AutoencoderVisualizer:
    def __init__(self, model_classes, model_paths, dataset, device=None):
        """
        Initialize the visualizer with multiple models, dataset, and device.
        
        :param model_classes: List of model classes (one for each model).
        :param model_paths: List of paths to the pre-trained model weights (one for each model).
        :param dataset: The dataset to use for visualization.
        :param device: The device to run the models on ('cuda' or 'cpu').
        """
        self.models = []
        self.models_name = []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        # Initialize and load each model
        for model_class, model_path in zip(model_classes, model_paths):
            model = model_class()
            model.to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.models.append(model)
            model_name = model_path.split('./results/')[1].split('.pt')[0]
            self.models_name.append(model_name)

    def visualize_comparisons(self, rows=5, output_path="multi_model_comparison.png"):
        """
        Visualize input images and reconstructions from multiple models in one figure.
        
        :param rows: Number of rows of input-output pairs to display.
        :param output_path: Path to save the combined visualization as a .png file.
        """
        cols = len(self.models) + 1  # 1 column for input + 1 for each model
        figure = plt.figure(figsize=(cols * 5, rows * 5))  # Adjust the figure size dynamically

        for i in range(rows):
            # Randomly select an image from the dataset
            index = torch.randint(len(self.dataset), size=(1,)).item()
            image, _ = self.dataset[index]
            image = image.unsqueeze(0).to(self.device)

            # Display input image (1st column)
            figure.add_subplot(rows, cols, i * cols + 1)
            plt.axis("off")
            if i == 0:  # Add column titles in the first row
                plt.title("Input")
            plt.imshow(image.squeeze().cpu(), cmap="Greys_r")

            # Display reconstructed images from each model
            for j, model in enumerate(self.models):
                with torch.no_grad():
                    output = model(image)

                figure.add_subplot(rows, cols, i * cols + j + 2)
                plt.axis("off")
                if i == 0:  # Add column titles in the first row
                    plt.title(self.models_name[j])
                plt.imshow(output.squeeze().cpu(), cmap="Greys_r")

        # Save the entire figure
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")
        plt.close(figure)  # Close the figure to free memory
