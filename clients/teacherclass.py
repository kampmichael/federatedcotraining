import torch
import torch.nn as nn

class teacher_class:
    def __init__(self):
        from papernet import FashionMNISTnet
        self.model = FashionMNISTnet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, X, y):
        self.model.train()
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            output = self.model(X)
        return output.softmax(dim=1).numpy()
