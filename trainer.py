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
        self.list_train_accuracy = []
        self.list_test_train_set_accuracy = []
        self.list_test_eval_set_accuracy = []

    def train_loop(self, dataloader):
        self.model.train()
        running_loss = 0
        sum_correct = 0

        for X, Y in dataloader:
            X = X.to(self.device)
            Y = Y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            sum_correct += (pred.argmax(1) == Y).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = sum_correct / len(dataloader.dataset)
        return avg_loss, accuracy

    def eval_loop(self, dataloader):
        self.model.eval()
        running_loss = 0
        sum_correct = 0
        
        with torch.no_grad():
            for X, Y in dataloader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)

                running_loss += loss.item()
                sum_correct += (pred.argmax(1) == Y).sum().item()

        avg_loss = running_loss / len(dataloader)
        accuracy = sum_correct / len(dataloader.dataset)

        return avg_loss, accuracy

    def train_model(self):
        for _ in tqdm(range(self.epochs)):
            train_loss, train_accuracy = self.train_loop(self.train_dataloader)
            test_train_set_loss, test_train_set_accuracy = self.eval_loop(self.train_dataloader)
            test_eval_set_loss, test_eval_set_accuracy = self.eval_loop(self.test_dataloader)

            self.list_train_losses.append(train_loss)
            self.list_test_train_set_losses.append(test_train_set_loss)
            self.list_test_eval_set_losses.append(test_eval_set_loss)
            self.list_train_accuracy.append(train_accuracy)
            self.list_test_train_set_accuracy.append(test_train_set_accuracy)
            self.list_test_eval_set_accuracy.append(test_eval_set_accuracy)
        return self.list_train_losses, self.list_test_train_set_losses, self.list_test_eval_set_losses, self.list_train_accuracy, self.list_test_train_set_accuracy, self.list_test_eval_set_accuracy


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
    def __init__(self, model ,config, t_dataset, v_dataset, device):
        self.config         = config
        self.t_dataset      = t_dataset
        self.v_dataset      = v_dataset
        self.model          = model.to(device)
        self.optimizer      = torch.optim.Adam((self.model.parameters() if config["train_all"] else self.model.classifier.parameters()), lr=config["learning_rate"])
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.train_loader   = torch.utils.data.DataLoader(t_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count()-1)
        self.valid_loader   = torch.utils.data.DataLoader(v_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count()-1)
        self.trainer        = Trainer(self.model, self.train_loader, self.valid_loader, self.loss_fn, self.optimizer, device, config["epochs"])

    def train(self):
        return self.trainer.train_model()

    def save_results(self, train_loss_incorrect, train_loss, valid_loss, train_accuracy_incorrect, train_accuracy, valid_accuracy):
        results = {
            "config": self.config,
            "train_loss_incorrect": train_loss_incorrect,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_accuracy_incorrect": train_accuracy_incorrect,
            "train_accuracy": train_accuracy,
            "valid_accuracy": valid_accuracy,
        }
        if not os.path.exists('./results_class'):
            os.makedirs('./results_class')
        file_name = f'./results_class/Classifier{"S2F" if self.config["S2F"] else "F2S"}_{"all" if self.config["train_all"] else "lineal"}'
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
