from torch import nn 
import torch

class LiveTrading(nn.Module):

    def __init__(self,lstm_input_size,lstm_hidden_size,num_layer,bidirectional,linear_hidden_size):

        super().__init__()

        self.lstm=nn.LSTM(lstm_input_size,lstm_hidden_size,num_layer,bidirectional)

        self.seq=nn.Sequential(
            nn.Linear(lstm_hidden_size,linear_hidden_size),
            nn.Tanh(),
            nn.Linear(linear_hidden_size,2*linear_hidden_size),
            nn.Tanh(),
            nn.Linear(2*linear_hidden_size,1),
            nn.ReLU()
        )
    def forward(self,x):

        o,(o1,o2)=self.lstm(x)
        op=self.seq(o)

        return op