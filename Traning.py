
import time

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
import networkx as nx
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
from Model import RNN1
from Model import RNN2
from Model import LSTM1
from Model import LSTM2
from Model import Classifier
from Data_Preprocessing import obtain_data1
from Data_Preprocessing import obtain_data2
from Data_Preprocessing import obtain_data3




def train(data, train_mask, test_mask, labels, model):
    def evaluate(data, labels, test_mask):
        model.eval()
        with torch.no_grad():
            logits = model(data)
            logits = logits[test_mask]
            labels = labels[test_mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    accuracy = []
    dur = []
    for e in range(4000):
        if e >=3:
            t0 = time.time()
        # Forward
        pred = model(data)
        # Compute prediction


        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        #loss = F.nll_loss(pred[train_mask],labels[train_mask])
        loss = F.cross_entropy(pred[train_mask],labels[train_mask])
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e >=3:
            dur.append(time.time() - t0)

        acc = evaluate(data, labels, test_mask)
        accuracy.append(acc)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(e, loss.item(), acc, np.mean(dur)))
        #print('Epoch %d | Loss: %.4f' % (e, loss.item()))
    return accuracy

Model = LSTM1()
data, label, train_mask, test_mask = obtain_data1()








train(data, train_mask, test_mask, label, Model)








