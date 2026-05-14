from model import DF, AWF, WF

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import wandb
import argparse
import pickle


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



def categorize(labels, dict_labels=None):
    possible_labels = list(set(labels))
    possible_labels.sort()
    print(possible_labels)
    if not dict_labels:
        dict_labels = {}
        n = 0
        for label in possible_labels:
            dict_labels[label] = n
            n = n + 1
        print("label_max: ",n)
        #print(dict_labels)
    new_labels = []
    for label in labels:
        new_labels.append(dict_labels[label])

    return new_labels



class WFDataset(Dataset):
    def __init__(self, data , label):
        self.raw_data = np.array(data).astype(dtype="float_")[:,:,0]

        self.data = np.transpose(self.raw_data.reshape(self.raw_data.shape[0], self.raw_data.shape[1], 1), (0, 2, 1))
        self.label = categorize(np.array(label))
        self.data_len = len(self.data)
        print("data load finish")

    def __getitem__(self, index):
        single_label = self.label[index]
        return (self.data[index], single_label)
 
    def __len__(self):
        return self.data_len


parse = argparse.ArgumentParser()
parse.add_argument('--LR',type=float,default=0.0001)
parse.add_argument('--BATCH_SIZE',type=int,default=512)
parse.add_argument('--gpu',type=int,default=None)
parse.add_argument('--epochs',type=int,default=300)
args = parse.parse_args()
BATCH_SIZE = args.BATCH_SIZE
LR = args.LR
if args.gpu is not None:
    torch.cuda.set_device(args.gpu)

# NB_CLASSES = 279
NB_CLASSES = 100
EPOCH = args.epochs

# wandb.init()
path1 = "./data/data_2000.pkl"
path2 = "./data/label_2000.pkl"


data = pickle.load(open(path1, "rb"))
label = pickle.load(open(path2, "rb"))


full_data = WFDataset(data,label)

train_split= 0.9
validate_split = 0.05
test_split = 0.05
shuffle_dataset = True
random_seed = 16
dataset_size = len(full_data)
print(dataset_size)
indices = list(range(dataset_size))
train_size = int(train_split * dataset_size)
validation_size = int(validate_split * dataset_size)
test_size = int(dataset_size - train_size - validation_size)
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices= indices[:train_size], indices[train_size:train_size+validation_size], indices[train_size+validation_size:]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(full_data, batch_size=BATCH_SIZE, 
                                            sampler=train_sampler)
validation_loader = DataLoader(full_data, batch_size=BATCH_SIZE,
                                            sampler=valid_sampler)
test_loader = DataLoader(full_data, batch_size=BATCH_SIZE,
                                            sampler=test_sampler)

cuda_gpu = torch.cuda.is_available()
cnn = DF(NB_CLASSES).float().cuda()
# if(cuda_gpu):
#     cnn = torch.nn.DataParallel(cnn, device_ids=[0]).cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):

        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = cnn(b_x.float())[0]
        # print(output)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # exit(0)
        if step % 50 == 0:
            corrects = 0
            avg_loss = 0
            for _, (b_x, b_y) in enumerate(validation_loader):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                logit = cnn(b_x.float())[0]
                loss = loss_func(logit, b_y)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)
                            [1].view(b_y.size()).data == b_y.data).sum()

            size = validation_size
            avg_loss /= size
            accuracy = 100.0 * corrects / size

            # wandb.log({"avg_loss":avg_loss,'accuracy':accuracy})
            print('Epoch: {:2d}({:6d}/{}) Evaluation - loss: {:.6f}  acc: {:3.4f}%({}/{})'.format(
                                                                            epoch,
                                                                            step * 128,
                                                                            train_size,
                                                                            avg_loss,
                                                                            accuracy,
                                                                            corrects,
                                                                            size))

torch.save(cnn.state_dict(), './model_DF.pth')

corrects = 0
avg_loss = 0
for _, (b_x, b_y) in enumerate(test_loader):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                logit = cnn(b_x.float())[0]
                loss = loss_func(logit, b_y)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)
                            [1].view(b_y.size()).data == b_y.data).sum()

size = test_size
accuracy = 100.0 * corrects / size
print("accuracy: {:3.4f}%".format(accuracy))

# wandb.finish()
# Accuracy is around 89% on 279 classes.
