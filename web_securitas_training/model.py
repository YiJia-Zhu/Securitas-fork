import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class DF(nn.Module):
    def __init__(self, nb_classes):
        super(DF, self).__init__()
        self.block1 = nn.Sequential(         
            nn.Conv1d(
                in_channels=1,              
                out_channels=32,            
                kernel_size=8,              
                stride=1,                   
                padding=0,                 
            ),  
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),                     
            nn.Conv1d(32, 32, 8, 1, 0),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(8, 4, 0), 
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 8, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 8, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, 1, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 8, 1, 0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 8, 1, 0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1),
        )

        self.fc1 = nn.Sequential(         
            nn.Flatten(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.out = nn.Sequential(
            nn.Linear(256, nb_classes),
            nn.Softmax(dim=1)
        )   

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.fc1(x)
        x = self.fc2(x)          
        output = self.out(x)
        return output, x   


class AWF(nn.Module):
    def __init__(self, nb_classes, dropout):
        super(AWF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout),         
            nn.Conv1d(1, 32, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(4), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*123, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, nb_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        output = self.out(x)
        return output, x


class WF(nn.Module):
    def __init__(self, nb_classes):
        super(WF, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 32, 8, 1, 0),
            nn.BatchNorm1d(32), 
            nn.ELU(),                     
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1), 
        )

        self.conv2 = nn.Sequential(       
            nn.Conv1d(32, 64, 8, 1, 0),
            nn.BatchNorm1d(64), 
            nn.ELU(),                     
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1), 
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(19776, 500),
            nn.ReLU(),
            nn.Linear(500, nb_classes)
        )   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)     
        output = self.out(x)
        return output, x  


def extract(sinste):
    
    # sinste: list of packet sizes

    # first 4 features
    sinste = np.array(sinste)
    sinste = sinste[:,1]*sinste[:,2]


    insize = 0
    outsize = 0
    inpacket = 0
    outpacket = 0


    n = 100  # number of linear interpolants

    x = 0  # sum of packet sizes
    y = 0  # sum of absolute packet sizes
    graph = []

    for i in range(0, len(sinste)):
        if sinste[i] > 0:
            outsize += sinste[i]
            outpacket += 1
        else:
            insize += -1*sinste[i]
            inpacket += 1

        x += abs(sinste[i])
        y += sinste[i]
        graph.append([x, y])

    features = [insize, outsize, inpacket, outpacket]
    # features = []


    # 100 interpolants

    # derive interpolants
    max_x = graph[-1][0]  # sum of absolute values
    gap = float(max_x) / n
    cur_x = 0
    cur_y = 0
    graph_ptr = 0

    for i in range(0, n):
        # take linear line between cur_x and cur_x + gap
        next_x = cur_x + gap
        while (graph[graph_ptr][0] < next_x):
            graph_ptr += 1
            if (graph_ptr >= len(graph) - 1):
                graph_ptr = len(graph) - 1
                # wouldn't be necessary if floats were floats
                break
        ##        print "graph_ptr=", graph_ptr
        next_pt_y = graph[graph_ptr][1]  # not next_y
        next_pt_x = graph[graph_ptr][0]
        cur_pt_y = graph[graph_ptr - 1][1]
        cur_pt_x = graph[graph_ptr - 1][0]
        ##        print "lines are", cur_pt_x, cur_pt_y, next_pt_x, next_pt_y

        if (next_pt_x - cur_pt_x != 0):
            slope = (next_pt_y - cur_pt_y) / (next_pt_x - cur_pt_x)
        else:
            slope = 1000
        next_y = slope * (next_x - cur_pt_x) + cur_pt_y

        interpolant = (next_y - cur_y) / (next_x - cur_x)
        features.append(interpolant)
        cur_x = next_x
        cur_y = next_y
    return features

def categorize(labels, dict_labels=None):
    possible_labels = list(set(np.array(labels)))
    possible_labels.sort()
    if not dict_labels:
        dict_labels = {}
        n = 0
        for label in possible_labels:
            dict_labels[label] = n
            n = n + 1
        #print("label_max: ",n)
    new_labels = []
    for label in labels:
        new_labels.append(dict_labels[label])

    return new_labels

def normalize_data(data):
    # Compute the maximum and minimum value for each column.
    max_vals = np.max(data, axis=0)
    min_vals = np.min(data, axis=0)

    # Scale the data to the [-1, 1] range.
    normalized_data = -1 + 2 * (data - min_vals) / (max_vals - min_vals)

    return normalized_data

def CUMUL(data, label, svm):
    X= []
    for row in data:
        X.append(extract(row))

    # npX=np.array(X)

    X_test = normalize_data(X)
    y_test = label

    y_pred = svm.predict(X_test)
    y_pred_pro = svm.predict_proba(X_test)
    y_pred = np.argmax(y_pred_pro,axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred_pro


class DilatedBasicBlock1D(nn.Module):
    def __init__(self, input_dim,filters, layer, block, dilations):
        super(DilatedBasicBlock1D, self).__init__()
        self.label = f'layer{layer}_block{block}'

        if layer == 2 or block != 1:
            self.stride = 1
        else:
            self.stride = 2


        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=3, stride=self.stride,
                               padding=dilations[0], dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(filters, filters, kernel_size=3, padding=dilations[1],
                               dilation=dilations[1], bias=False)
        self.bn2 = nn.BatchNorm1d(filters)

        if layer > 2 and block == 1:
            self.ds_conv = nn.Conv1d(input_dim, filters, kernel_size=1, stride=self.stride, bias=False)
            self.ds_bn = nn.BatchNorm1d(filters)

    def forward(self, x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if hasattr(self, 'ds_conv'):
            shortcut = self.ds_conv(x)
            shortcut = self.ds_bn(shortcut)
        else:
            shortcut = x

        y += shortcut
        y = F.relu(y)
        return y

class VarCNN(nn.Module):
    def __init__(self, classes):
        super(VarCNN, self).__init__()

        self.layer_blocks = [2, 2, 2, 2]

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        features = 32
        input_dim = 64
        self.layers = nn.ModuleList()
        for i, blocks in enumerate(self.layer_blocks):
            features *= 2

            self.layers.append(DilatedBasicBlock1D(input_dim,features, i+2, 1, dilations=(1, 2)))
            for block in range(2, blocks+1):
                self.layers.append(DilatedBasicBlock1D(features,features, i+2, block, dilations=(4, 8)))
            
            if i==0:
                input_dim = 64
            else:
                input_dim*=2
            

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(features, classes)
        
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.softmax_layer(x)

        return x, 0
