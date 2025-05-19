import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class HybridCNN_MLP_Bayesian(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(HybridCNN_MLP_Bayesian, self).__init__()

        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # MLP branch
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(32 * (input_dim // 4) + 64, 64),  
            nn.ReLU(),
            nn.Dropout(0.7),  
            nn.Linear(64, num_classes)
        )

    def forward(self, x_raw, x_seq):
        cnn_out = self.cnn(x_seq)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        mlp_out = self.mlp(x_raw)
        combined = torch.cat((cnn_out, mlp_out), dim=1)
        return self.classifier(combined)