import torch
from tqdm import tqdm
import os
import json
import autoencoders

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs=30):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.list_train_losses = []
        self.list_test_train_set_losses = []
        self.list_test_eval_set_losses = []

    def train_loop(self):
        self.model.train()
        running_loss = 0
        for (X, Y) in self.train_dataloader:
            X = X.to(self.device)
            Y = Y.to(self.device)
            output = self.model(X)
            loss = self.loss_fn(output, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(self.train_dataloader)
        return avg_loss

    def eval_loop(self, dataloader):
        self.model.eval()
        running_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for X, Y in dataloader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)

                running_loss += loss.item()

        avg_loss = running_loss / num_batches
        return avg_loss

    def train_model(self):
        for _ in tqdm(range(self.epochs)):
            train_loss = self.train_loop()
            test_train_set_loss = self.eval_loop(self.train_dataloader)
            test_eval_set_loss = self.eval_loop(self.test_dataloader)

            self.list_train_losses.append(train_loss)
            self.list_test_train_set_losses.append(test_train_set_loss)
            self.list_test_eval_set_losses.append(test_eval_set_loss)
        return self.list_train_losses, self.list_test_train_set_losses, self.list_test_eval_set_losses


class AutoencoderTrainer:
    def __init__(self, config, t_dataset, v_dataset):
        self.config = config
        self.t_dataset = t_dataset
        self.v_dataset = v_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.loss_fn = torch.nn.MSELoss()
        self.train_loader = torch.utils.data.DataLoader(t_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count()-1)
        self.valid_loader = torch.utils.data.DataLoader(v_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count()-1)
        self.trainer = Trainer(self.model, self.train_loader, self.valid_loader, self.loss_fn, self.optimizer, self.device, config["epochs"])

    def _create_model(self):
        if self.config["S2F"]:
            return autoencoders.Autoencoder_S2F(self.config["dropout"]).to(self.device)
        else:
            return autoencoders.Autoencoder_F2S(self.config["dropout"]).to(self.device)

    def train(self):
        train_loss_incorrect, train_loss, valid_loss = self.trainer.train_model()
        return train_loss_incorrect, train_loss, valid_loss

    def save_results(self, train_loss_incorrect, train_loss, valid_loss):
        results = {
            "config": self.config,
            "train_loss_incorrect": train_loss_incorrect,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        }
        if not os.path.exists('./results'):
            os.makedirs('./results')
        file_name = f'./results/{"S2F" if self.config["S2F"] else "F2S"}_{self.config["learning_rate"]}'
        torch.save(self.model.state_dict(), f'{file_name}.pt')
        with open(f'{file_name}.json', 'w') as f:
            json.dump(results, f)


class ConvClassifierTrainer:
    def __init__(self, config, t_dataset, v_dataset):
        self.config = config
        self.t_dataset = t_dataset
        self.v_dataset = v_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = autoencoders.Classifier().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_loader = torch.utils.data.DataLoader(t_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count()-1)
        self.valid_loader = torch.utils.data.DataLoader(v_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count()-1)
        self.trainer = Trainer(self.model, self.train_loader, self.valid_loader, self.loss_fn, self.optimizer, self.device, config["epochs"])

    def train(self):
        train_loss_incorrect, train_loss, valid_loss = self.trainer.train_model()
        return train_loss_incorrect, train_loss, valid_loss

    def save_results(self, train_loss_incorrect, train_loss, valid_loss):
        results = {
            "config": self.config,
            "train_loss_incorrect": train_loss_incorrect,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        }
        if not os.path.exists('./results'):
            os.makedirs('./results')
        file_name = f'./results/Classifier_{self.config["learning_rate"]}'
        torch.save(self.model.state_dict(), f'{file_name}.pt')
        with open(f'{file_name}.json', 'w') as f:
            json.dump(results, f)

class AutoencoderTrainingManager:
    def __init__(self, configs, t_dataset, v_dataset):
        self.configs = configs
        self.t_dataset = t_dataset
        self.v_dataset = v_dataset

    def train_all(self):
        results = []
        for config in self.configs:
            trainer = AutoencoderTrainer(config, self.t_dataset, self.v_dataset)
            train_loss_incorrect, train_loss, valid_loss = trainer.train()
            trainer.save_results(train_loss_incorrect, train_loss, valid_loss)
            results.append((config, train_loss_incorrect, train_loss, valid_loss))
        return results
