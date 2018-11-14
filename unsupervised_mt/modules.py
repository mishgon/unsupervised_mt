import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, embedding=nn.Embedding):
        super(Encoder, self).__init__()
        if not embedding:
            self.embedding = nn.Embedding(input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional)