import torch
from tqdm import tqdm

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
            output = self.model(X)
            loss = self.loss_fn(output, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        return running_loss / len(self.train_dataloader)

    def eval_loop(self, dataloader):
        self.model.eval()
        running_loss = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for X, Y in dataloader:
                X = X.to(self.device)
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