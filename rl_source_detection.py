import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class PolicyNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n, 128),
            nn.ReLU(),
            nn.Linear(128, n),
        )

    def forward(self, state_mask):
        logits = self.fc(state_mask)
        logits[state_mask.bool()] = -1e9  # mask already selected
        return torch.softmax(logits, dim=-1)

def reinforce_train(f, V, k, epochs=1000, lr=1e-2):
    n = len(V)
    policy = PolicyNet(n)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    for epoch in range(epochs):
        S = []
        log_probs = []
        state_mask = torch.zeros(n)

        for _ in range(k):
            probs = policy(state_mask)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            S.append(V[action.item()])
            state_mask[action.item()] = 1

        reward = f(set(S))  # Submodular function evaluation
        loss = -sum(log_probs) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: reward = {reward:.4f}")

    return policy
