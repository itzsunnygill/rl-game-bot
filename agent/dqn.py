import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),  # input layer
            nn.ReLU(),                    # activation
            nn.Linear(128, 128),          # hidden layer
            nn.ReLU(),                    # activation
            nn.Linear(128, output_size)   # output layer
        )

    def forward(self, x):
        return self.network(x)