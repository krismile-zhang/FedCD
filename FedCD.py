
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.optimize import minimize
import numpy as np



class FedCD_LocalUpdate(object):
    def __init__(self, idx, args, train_loader, test_loader, model):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_model = model
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.lr = args.lr
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=args.lr, momentum=0.5)

    def knowledge_distillation_loss(self, logits, label_logits):
        probs = F.softmax(logits, dim=1)

        label_probs = F.softmax(label_logits, dim=1)

        log_probs = torch.log(probs + 1e-9)
        kd_loss = torch.kl_div(log_probs, label_probs, reduction='batchmean')

        return kd_loss

    def get_all_logits(self, clients, features):
        all_logits = []
        num_clients = len(clients)
        for client in clients:
            logit = client.get_logit(features)
            all_logits.append(logit)
        return all_logits



    def constraint(α):
        return np.sum(α) - 1

    def constraint_gradient(α):
        return np.ones_like(α)

    def calculate_client_influence(self, all_logits, labels):
        N = len(all_logits)
        initial_alpha = np.full(N, 1 / N)
        result = minimize(
            fun=lambda α: self.criterion(np.dot(α, all_logits), labels),
            x0=initial_alpha,
            method='L-BFGS-B',
            jac=True,
            constraints={'type': 'eq', 'fun': self.constraint, 'jac': self.constraint_gradient}
        )

        weighted_logits = F.softmax(np.dot(result.x, all_logits), dim=1)

        return result.x, weighted_logits


    def train(self,  clients, t):
        model = self.local_model
        model.train()
        adjusted_lr = self.lr / (1 + t)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        epoch_loss = []
        for epoch in range(self.args.local_epochs):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.local_model.zero_grad()
                features, logits = self.local_model(data)
                all_logits = self.get_all_logits(clients, features)
                all_logits.append(logits)
                alpha, label_logits = self.calculate_client_influence(all_logits, target)
                kd_loss = self.knowledge_distillation_loss(logits, label_logits)
                lab_loss = self.criterion(logits, target)
                loss = self.args.gamma * kd_loss + (1-self.args.gamma) * lab_loss
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        t += 1
        return model.state_dict(), alpha, epoch_loss



