
# Imports
import torch.nn as nn
from collections import OrderedDict

# Internal Imports
from modules.seq_linear import SeqLinear

class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size):
        """

        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', SeqLinear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(0.5)),
             ('fc2', SeqLinear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(0.5)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out