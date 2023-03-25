
# Imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from collections import OrderedDict

# Internal Imports
from modules.seq_linear import SeqLinear


use_cuda = torch.cuda.is_available()

try: # older versions of torch don't recognise mps
    use_mps = torch.backends.mps.is_available() & torch.backends.mps.is_built()
    device = torch.device('mps')
except:
    use_mps = False

class CBHG(nn.Module):
    """
    CBHG Module
    """
    def __init__(self, hidden_size, K=16, projection_size = 128, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        """

        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K
        if is_post:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size * 2,
                                             kernel_size=3,
                                             padding=int(np.floor(3/2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size * 2,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3/2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size * 2)

        else:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)


        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size, num_layers=2,
                          batch_first=True,
                          bidirectional=True)


    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x

    def forward(self, input_):

        input_ = input_.contiguous()
        batch_size = input_.size()[0]

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = torch.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        # Projection
        conv_projection = torch.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_

        # Highway networks
        highway = self.highway.forward(conv_projection)
        highway = torch.transpose(highway, 1,2)

        # Bidirectional GRU
        if use_mps:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size).to(device=device))
        elif use_cuda:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size)).cuda()
        else:
            init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size))

        self.gru.flatten_parameters()
        out, _ = self.gru(highway, init_gru)

        return out


class Highwaynet(nn.Module):
    """
    Highway network
    """
    def __init__(self, num_units, num_layers=4):
        """

        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(SeqLinear(num_units, num_units))
            self.gates.append(SeqLinear(num_units, num_units))

    def forward(self, input_):

        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = torch.relu(fc1.forward(out))
            t = torch.sigmoid(fc2.forward(out))

            c = 1. - t
            out = h * t + out * c

        return out