
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from matplotlib import pyplot as plt

from Data_Preprocessing import obtain_data1


class RNN1(nn.Module):
    def __init__(self):
        super(RNN1, self).__init__()
        self.rnn_layer = nn.RNN(input_size=3, hidden_size=3, num_layers=1, batch_first=True)
        self.out_layer = nn.Linear(3,2)

    def forward(self, x):
        output, h_n = self.rnn_layer(x,None)
        out = self.out_layer(h_n[0])
        out = F.softmax(out,1)
        return out



class RNN2(nn.Module):
    def __init__(self):
        super(RNN2, self).__init__()
        self.rnn_layer = nn.RNN(input_size=6, hidden_size=6, num_layers=1, batch_first=True)
        self.out_layer1 = nn.Linear(6,4)
        self.out_layer2 = nn.Linear(4,2)

    def forward(self, x):
        output, h_n = self.rnn_layer(x,None)
        out = self.out_layer1(h_n[0])
        out = self.out_layer2(out)
        out = F.softmax(out,1)
        return out




class LSTM1(nn.Module):
    def __init__(self):
        super(LSTM1, self).__init__()
        self.rnn_layer = nn.LSTM(input_size=3, hidden_size=3, num_layers=1, batch_first=True)
        self.out_layer = nn.Linear(3,2)

    def forward(self, x):
        output, (hn, cn) = self.rnn_layer(x)
        out = self.out_layer(hn[0])
        out = F.softmax(out,1)
        return out

class Classifier(nn.Module):
    def __init__(self, in_feats, h_feats1, num_classes):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(in_features=in_feats, out_features=h_feats1, bias = True)
        self.last = nn.Linear(in_features=h_feats1, out_features=num_classes, bias=False)

    def forward(self, h):
        h = self.layer1(h)
        h = self.last(h)
        return h

class LSTM2(nn.Module):
    def __init__(self):
        super(LSTM2, self).__init__()
        self.rnn_layer = nn.LSTM(input_size=6, hidden_size=6, num_layers=1, batch_first=True)
        self.out_layer1 = nn.Linear(6,4)
        self.out_layer2 = nn.Linear(4,2)

    def forward(self, x):
        output, (hn, cn) = self.rnn_layer(x)
        out = self.out_layer1(hn[0])
        out = self.out_layer2(out)
        out = F.softmax(out,1)
        return out

