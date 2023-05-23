import numpy as np
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class CSCDataset(Dataset):
    def __init__(self, X, C):
        assert X.size()[0] == C.size()[0]
        self.X = X
        self.C = C

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.C[idx]]

class EBDataset(Dataset):
    def __init__(self, X, C, loss_shift):
        assert X.size()[0] == C.size()[0] == loss_shift.size()[0]
        self.X = X
        self.C = C
        self.loss_shift = loss_shift

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.C[idx], self.loss_shift[idx]]


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, X):
        out = self.linear(X)
        return out


class LinearRegressionCSCOracle:
    def __init__(self, weight_decay=1e-3, batch_size=100, epochs=1, lr=0.01, loss_type="l2"):
        """
        param weight_decay: float
                    L2 regularization strength
        param batch_size: int
        param epochs: int
        param lr: float
                    learning rate
        """
        self.weight_decay = weight_decay
        self.model = None
        self.best_model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.loss_type = loss_type

    def fit(self, X_npy, C_npy, opt_type="SGD"):
        """
        param X: numpy array (n_examples, n_features)
                    features of the examples
        param C: numpy array (n_examples, n_classes)
                    costs of the examples
        return: cost sensitive classification model
        """
        X = torch.from_numpy(X_npy).float()
        C = torch.from_numpy(C_npy).float()
        self.model = LinearModel(X.size()[1], C.size()[1])
        if opt_type == "SGD":
            data = DataLoader(CSCDataset(X, C), batch_size=self.batch_size, shuffle=True)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            # best_loss = None
            for epoch in range(self.epochs):
                # total_loss = self.loss(X_npy, C_npy)
                # if best_loss is None or total_loss < best_loss:
                #     self.best_model = copy.deepcopy(self.model)

                for i, (X_batch, C_batch) in enumerate(data):
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    if self.loss_type == "l2":
                        loss = ((outputs - C_batch) ** 2).mean()
                    elif self.loss_type == "l1":
                        loss = torch.abs(outputs - C_batch).mean()
                    else:
                        raise ValueError
                    loss.backward()
                    optimizer.step()
        elif opt_type == "LBFGS":
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
            # best_loss = None
            for epoch in range(self.epochs):
                # total_loss = self.loss(X_npy, C_npy)
                # if best_loss is None or total_loss < best_loss:
                #     self.best_model = copy.deepcopy(self.model)

                def closure():
                    optimizer.zero_grad()
                    outputs = self.model(X)
                    if self.loss_type == "l2":
                        loss = ((outputs - C) ** 2).mean()
                    elif self.loss_type == "l1":
                        loss = torch.abs(outputs - C).mean()
                    else:
                        raise ValueError
                    l2_reg = sum(torch.sum(param * param) for param in self.model.parameters())
                    loss = loss + self.weight_decay * l2_reg
                    loss.backward()
                    return loss
                optimizer.step(closure)
                # if not ((i+1) % 200):
                # print("Epoch: [%d/%d] Loss: %.3f" % (epoch + 1, self.epochs, total_loss))

        else:
            raise ValueError
        # self.model = self.best_model
        return self.model

    def loss(self, X, C):
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = self.model(X)
            if self.loss_type == "l2":
                loss = ((outputs - C) ** 2).mean()
            elif self.loss_type == "l1":
                loss = torch.abs(outputs - C).mean()
            else:
                raise ValueError
        return loss.item()

    def predict(self, X):
        """
        param X: numpy array (n_examples, n_features)
                    features of the examples
        return: C_hat: one hot representation of class prediction
        """
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = self.model(X).numpy()
        predictions = np.argmin(outputs, axis=1)
        return predictions

    def predict_score(self, X):
        """
        param X: numpy array (n_examples, n_features)
                    features of the examples
        return: scores: numpy array (n_examples, n_actions)
        """
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = self.model(X).numpy()
        return outputs


class LinearSoftmaxCSCOracle:
    def __init__(self, weight_decay=1e-3, batch_size=100, epochs=1, lr=0.01):
        """
        param weight_decay: float
                    L2 regularization strength
        param batch_size: int
        param epochs: int
        param lr: float
                    learning rate
        """
        self.weight_decay = weight_decay
        self.model = None
        self.best_model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def fit(self, X_npy, C_npy, opt_type="SGD"):
        """
        param X: numpy array (n_examples, n_features)
                    features of the examples
        param C: numpy array (n_examples, n_classes)
                    costs of the examples
        return: cost sensitive classification model
        """
        X = torch.from_numpy(X_npy).float()
        C = torch.from_numpy(C_npy).float()
        self.model = LinearModel(X.size()[1], C.size()[1])
        m = nn.Softmax(dim=1)
        if opt_type == "SGD":
            data = DataLoader(CSCDataset(X, C), batch_size=self.batch_size, shuffle=True)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            best_loss = None
            for epoch in range(self.epochs):
                # total_loss = self.loss(X_npy, C_npy)
                # if best_loss is None or total_loss < best_loss:
                #     self.best_model = copy.deepcopy(self.model)
                for i, (X_batch, C_batch) in enumerate(data):
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = (m(outputs) * C_batch).mean() * C.size()[1]
                    loss.backward()
                    optimizer.step()
        elif opt_type == "LBFGS":
            optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
            # best_loss = None
            for epoch in range(self.epochs):
                # total_loss = self.loss(X_npy, C_npy)
                # if best_loss is None or total_loss < best_loss:
                #     self.best_model = copy.deepcopy(self.model)

                def closure():
                    optimizer.zero_grad()
                    outputs = self.model(X)
                    loss = (m(outputs) * C).mean() * C.size()[1]
                    l2_reg = sum(torch.sum(param * param) for param in self.model.parameters())
                    loss = loss + self.weight_decay * l2_reg
                    loss.backward()
                    return loss

                optimizer.step(closure)
                # if not ((i+1) % 200):
                # print("Epoch: [%d/%d] Loss: %.3f" % (epoch + 1, self.epochs, total_loss))
        # self.model = self.best_model
        return self.model

    def loss(self, X, C):
        X = torch.from_numpy(X).float()
        C = torch.from_numpy(C).float()
        m = nn.Softmax(dim=1)
        with torch.no_grad():
            outputs = self.model(X)
            loss = (m(outputs) * C).mean() * C.size()[1]
        return loss.item()

    def predict(self, X):
        """
        param X: numpy array (n_examples, n_features)
                    features of the examples
        return: C_hat: one hot representation of class prediction
        """
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = self.model(X).numpy()
        predictions = np.argmax(outputs, axis=1)
        return predictions


class LinearSoftmaxEBOracle:
    def __init__(self, weight_decay=1e-3, batch_size=100, epochs=10, lr=0.01):
        """
        param weight_decay: float
                    L2 regularization strength
        param epochs: int
        param lr: float
                    learning rate
        """
        self.weight_decay = weight_decay
        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.best_model = None

    def fit(self, X_npy, C_npy, loss_shift_npy, beta, opt_type="LBFGS"):
        X = torch.from_numpy(X_npy).float()
        C = torch.from_numpy(C_npy).float()
        loss_shift = torch.from_numpy(loss_shift_npy).float()
        self.model = LinearModel(X.size()[1], C.size()[1])
        m = nn.Softmax(dim=1)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
        best_loss = None
        if opt_type == "LBFGS":
            for epoch in range(self.epochs):
                # total_loss = self.loss(X_npy, A_npy, L_npy, P_npy, beta)
                # if best_loss is None or total_loss < best_loss:
                #     self.best_model = copy.deepcopy(self.model)
                def closure():
                    optimizer.zero_grad()
                    outputs = self.model(X)
                    sample = torch.sum(m(outputs) * C, dim=1) + loss_shift
                    loss_IPW = torch.mean(sample)
                    sample_variance = torch.sum((sample - loss_IPW) * (sample - loss_IPW)) / (sample.size()[0] - 1)
                    l2_reg = sum(torch.sum(param * param) for param in self.model.parameters())
                    loss = loss_IPW + beta * torch.sqrt(sample_variance) + self.weight_decay * l2_reg
                    loss.backward()
                    return loss
                optimizer.step(closure)
                    # if not ((i+1) % 200):
                # print("Epoch: [%d/%d] Loss: %.3f" % (epoch + 1, self.epochs, total_loss))
            # self.model = self.best_model
        elif opt_type == "SGD":
            data = DataLoader(EBDataset(X, C, loss_shift), batch_size=self.batch_size, shuffle=True)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9,
                                        weight_decay=self.weight_decay)
            # best_loss = None
            for epoch in range(self.epochs):
                # total_loss = self.loss(X_npy, C_npy)
                # if best_loss is None or total_loss < best_loss:
                #     self.best_model = copy.deepcopy(self.model)
                for i, (X_batch, C_batch, loss_shift_batch) in enumerate(data):
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    sample = torch.sum(m(outputs) * C_batch, dim=1) + loss_shift_batch
                    loss_IPW = torch.mean(sample)
                    sample_variance = torch.sum((sample - loss_IPW) * (sample - loss_IPW)) / (sample.size()[0] - 1)
                    loss = loss_IPW + beta * torch.sqrt(sample_variance)
                    loss.backward()
                    optimizer.step()
        else:
            raise ValueError
        return self.model

    def loss(self, X, C, loss_shift, beta):
        X = torch.from_numpy(X).float()
        C = torch.from_numpy(C).float()
        loss_shift = torch.from_numpy(loss_shift).float()
        m = nn.Softmax(dim=1)
        with torch.no_grad():
            outputs = self.model(X)
            sample =torch.sum(m(outputs) * C, dim=1) + loss_shift
            loss_IPW = torch.mean(sample)
            sample_variance = torch.sum((sample - loss_IPW) * (sample - loss_IPW)) / (sample.size()[0] - 1)
            loss = loss_IPW + beta * torch.sqrt(sample_variance / (sample.size()[0]))
        return loss.item()

    def predict(self, X):
        """
        param X: numpy array (n_examples, n_features)
                    features of the examples
        return: C_hat: one hot representation of class prediction
        """
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = self.model(X).numpy()
        predictions = np.argmax(outputs, axis=1)
        return predictions






