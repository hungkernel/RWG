import torch
import torch.nn as nn
import numpy as np
from src.utils import layer_init
from src import config
from src.agent.action import NUM_ACTIONS

class Critic(nn.Module):
    """
    Critic network (value network) that takes vector observations and outputs state value.
    Input: concatenated observation vector (embed_dim * 4 + NUM_ACTIONS + 1 + embed_dim)
    Output: scalar state value
    """
    def __init__(self, obs_dim: int = None):
        super(Critic, self).__init__()
        # Calculate observation dimension: 4 embeddings + agent_state + score + 1 critique embedding
        if obs_dim is None:
            obs_dim = config.EMBED_DIM * 4 + NUM_ACTIONS + 1 + config.EMBED_DIM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Multi-layer perceptron
        self.fc1 = layer_init(nn.Linear(obs_dim, 256))
        self.relu1 = nn.ReLU()
        self.fc2 = layer_init(nn.Linear(256, 128))
        self.relu2 = nn.ReLU()
        self.fc3 = layer_init(nn.Linear(128, 1), std=1.0)

        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        value = self.fc3(x)
        return value