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
import requests
import csv



def build_Graph(data, L):
   Dict_Industry = {}
   Dict_City = {}
   Dict_Province = {}
   label = {}

   i = 1
   G = nx.Graph()

   for item in data:
      if(not (item[1] in L)):
          continue
      G.add_node(i)
      label[i] = item[1]
      if(item[8] in list(Dict_Industry.keys())):
          Dict_Industry[item[8]].append(i)
      else:
          Dict_Industry[item[8]] = [i]

      if(item[14] in list(Dict_Province.keys())):
          Dict_Province[item[14]].append(i)
      else:
          Dict_Province[item[14]] = [i]

      if(item[16] in list(Dict_City.keys())):
          Dict_City[item[16]].append(i)
      else:
          Dict_City[item[16]] = [i]

      i = i+1

   for item in list(Dict_Industry.keys()):
       for i in range(len(Dict_Industry[item])-1):
           for j in range(i+1, len(Dict_Industry[item])):
               if(Dict_Industry[item][j] in list(G.neighbors(Dict_Industry[item][i]))):
                   continue
               else:
                   G.add_edge(Dict_Industry[item][i], Dict_Industry[item][j], weight = 1)

   for item in list(Dict_Province.keys()):
       for i in range(len(Dict_Province[item])-1):
           for j in range(i+1, len(Dict_Province[item])):
               if(Dict_Province[item][j] in list(G.neighbors(Dict_Province[item][i]))):
                   G[Dict_Province[item][i]][Dict_Province[item][j]]['weight'] += 1
               else:
                   G.add_edge(Dict_Province[item][i], Dict_Province[item][j], weight = 1)

   for item in list(Dict_City.keys()):
       for i in range(len(Dict_City[item])-1):
           for j in range(i+1, len(Dict_City[item])):
                G[Dict_City[item][i]][Dict_City[item][j]]['weight'] += 1

   with open('Graph', 'wb') as f:
       pickle.dump(G, f)

   with open('label', 'wb') as f:
       pickle.dump(label, f)

def add_feature():
    with open('label2', 'rb') as file:
        label2 = pickle.load(file)

    with open('Graph', 'rb') as file:
        Graph = pickle.load(file)

    with open('Company_Code', 'rb') as file:
        code = pickle.load(file)

    data1 = pd.read_csv('Data2_.csv')
    data1 = data1.values.tolist()

    data2 = pd.read_csv('Table2_.csv')
    data2 = data2.values.tolist()

    j = 1
    for i in range(1,len(data1)):
        if(not (code[data1[i][3]] in list(label2.keys()))):
            continue
        if(j == 73):
            index = label2[code[data1[i][3]]]
            if(data1[i][5]>=0):
                Graph.nodes[index]['label'] = 1
            else:
                Graph.nodes[index]['label'] = 0
            j = 1
            continue
        index = label2[code[data1[i][3]]]
        Graph.nodes[index][j] = [data1[i][4]]
        Graph.nodes[index][j].append(data1[i][5])
        j = j+1

    j = 1
    for i in range(1,len(data2)):
        if(not(code[data1[i][3]] in list(label2.keys()))):
            continue
        if(j==73):
            j = 1
            continue
        index = label2[code[data2[i][3]]]
        Graph.nodes[index][j].append(data2[i][4])
        Graph.nodes[index][j].append(data2[i][5])
        j = j+1

    with open('Graph', 'wb') as f:
        pickle.dump(Graph, f)

def Message_Passing1():
    with open('Graph', 'rb') as file:
        G = pickle.load(file)

    new_G = G.copy(as_view=False)

    for item in list(G.nodes):
        total = 0
        for nodes in list(G.neighbors(item)):
            total = total + np.exp(G[item][nodes]['weight'])
        for i in range(1,73):
            Temp = np.array([0,0,0,0])
            for nodes in list(G.neighbors(item)):
                Temp = np.add(Temp, ((np.exp(G[item][nodes]['weight']))/total)*np.array(G.nodes[nodes][i]))
            new_G.nodes[item][i] = np.concatenate([Temp, np.array(G.nodes[item][i])])

    with open('Graph2', 'wb') as f:
        pickle.dump(new_G, f)

def Message_Passing2(alpha):
    with open('Graph', 'rb') as file:
        G = pickle.load(file)

    new_G = G.copy(as_view=False)

    for item in list(G.nodes):
        total = 0
        for nodes in list(G.neighbors(item)):
            total = total + np.exp(G[item][nodes]['weight'])
        for i in range(1,73):
            Temp = np.array([0,0,0,0])
            for nodes in list(G.neighbors(item)):
                Temp = np.add(Temp, ((np.exp(G[item][nodes]['weight']))/total)*np.array(G.nodes[nodes][i]))
            new_G.nodes[item][i] = np.add((1-alpha)*Temp, alpha*(np.array(G.nodes[item][i])))

    with open('Graph3', 'wb') as f:
        pickle.dump(new_G, f)

def obtain_data2():
    with open('Graph2', 'rb') as file:
        G2 = pickle.load(file)
    Tensor = []
    labels = []
    for i in range(1,225):
       L = []
       for j in range(1,73):
           L.append([G2.nodes[i][j][0], G2.nodes[i][j][1], G2.nodes[i][j][3], G2.nodes[i][j][4], G2.nodes[i][j][5], G2.nodes[i][j][7]])
       Tensor.append(L)
       labels.append(G2.nodes[i]['label'])

    #create training mask
    train_mask = [False]*224
    for i in range(200):
       train_mask[i] = True
    train_mask = torch.tensor(train_mask)

    #create testing mask
    test_mask = [False]*224
    for i in range(200,len(test_mask)):
      test_mask[i] = True
    test_mask = torch.tensor(test_mask)

    labels = torch.LongTensor(labels)
    Tensor = torch.Tensor(Tensor)

    return Tensor, labels, train_mask, test_mask

def obtain_data3():
    with open('Graph3', 'rb') as file:
        G3 = pickle.load(file)
    Tensor = []
    labels = []
    for i in range(1,225):
       L = []
       for j in range(1,73):
           L.append([G3.nodes[i][j][0], G3.nodes[i][j][1], G3.nodes[i][j][3]])
       Tensor.append(L)
       labels.append(G3.nodes[i]['label'])

    #create training mask
    train_mask = [False]*224
    for i in range(200):
       train_mask[i] = True
    train_mask = torch.tensor(train_mask)

    #create testing mask
    test_mask = [False]*224
    for i in range(200,len(test_mask)):
        test_mask[i] = True
    test_mask = torch.tensor(test_mask)

    labels = torch.LongTensor(labels)
    Tensor = torch.Tensor(Tensor)

    return Tensor, labels, train_mask, test_mask

def obtain_data1():
    with open('Graph', 'rb') as file:
        G1 = pickle.load(file)
    Tensor = []
    labels = []
    for i in range(1,225):
        L = []
        for j in range(1,73):
            L.append([G1.nodes[i][j][0], G1.nodes[i][j][1], G1.nodes[i][j][3]])
        Tensor.append(L)
        labels.append(G1.nodes[i]['label'])

    #create training mask
    train_mask = [False]*224
    for i in range(200):
        train_mask[i] = True
    train_mask = torch.tensor(train_mask)

    #create testing mask
    test_mask = [False]*224
    for i in range(200,len(test_mask)):
        test_mask[i] = True
    test_mask = torch.tensor(test_mask)

    labels = torch.LongTensor(labels)
    Tensor = torch.Tensor(Tensor)

    return Tensor, labels, train_mask, test_mask



Tensor, labels, train_mask, test_mask = obtain_data2()

"""
df = pd.DataFrame(data2)
df.to_csv('Data2_.csv')
"""

with open('Graph', 'rb') as file:
    Graph = pickle.load(file)

print(nx.diameter(Graph))















