import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_baseline(nn.Module):
    def __init__(self, output_class, embed_dim, kernel_num, kernel_size, dropout) -> None:
        super(CNN_baseline, self).__init__()
        self.conv_layer = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size)*kernel_num, output_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layer]
        x = [nn.MaxPool1d(i.size(2)) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


