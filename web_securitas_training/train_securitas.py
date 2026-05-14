from model import DF
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import time
from BSCAttack_agents import agent
from utils import test_epoch
import pickle
import argparse
import os
import random

NB_CLASSES = 100


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
        print("label_max: ", n)
    new_labels = []
    for label in labels:
        new_labels.append(dict_labels[label])

    return new_labels


def parse_loss_weights(value):
    weights = [float(x) for x in value.split(',')]
    if len(weights) != 3:
        raise argparse.ArgumentTypeError('loss weights must be three comma-separated floats')
    return weights


parse = argparse.ArgumentParser()
parse.add_argument('--model-name', choices=['DF'], default='DF')
parse.add_argument('--patch-length', type=int, default=800)
parse.add_argument('--epochs', type=int, default=10000)
parse.add_argument('--batch-size', type=int, default=256)
parse.add_argument('--lr', type=float, default=0.001)
parse.add_argument('--gpu', type=int, default=None)
parse.add_argument('--split-ratio', type=float, default=0.7)
parse.add_argument('--loss-weights', type=parse_loss_weights, default=[0.6, 0.2, 0.2])
parse.add_argument('--use-full-data', action='store_true')
parse.add_argument('--data-path', default='./data/data_2000.pkl')
parse.add_argument('--label-path', default='./data/label_2000.pkl')
args = parse.parse_args()

model_name = args.model_name
patch_length = args.patch_length
EPOCH = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
use_part_data = not args.use_full_data

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)

seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data = pickle.load(open(args.data_path, "rb"))
label = pickle.load(open(args.label_path, "rb"))

data = np.array(data).astype(np.float32)
label = categorize(np.array(label))

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=2023)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

if model_name == 'DF':
    model = DF(NB_CLASSES).float().cuda()
    model.load_state_dict(torch.load("./model_DF.pth", map_location='cpu'))

model = model.cuda()
model = model.eval()

train_loader = []
test_loader = []

if use_part_data:
    train_len = int(len(X_train) / 3)
    test_len = int(len(X_test) / 3)
else:
    train_len = len(X_train)
    test_len = len(X_test)

train_idx = 0
while train_idx < train_len:
    train_loader.append((X_train[train_idx:min(train_idx + BATCH_SIZE, train_len)][:, :, :, 0],
                         X_train[train_idx:min(train_idx + BATCH_SIZE, train_len)][:, :, :, 1],
                         Y_train[train_idx:min(train_idx + BATCH_SIZE, train_len)]))
    train_idx += BATCH_SIZE

test_idx = 0
while test_idx < test_len:
    test_loader.append((X_test[test_idx:min(test_idx + BATCH_SIZE, test_len)][:, :, :, 0],
                        X_test[test_idx:min(test_idx + BATCH_SIZE, test_len)][:, :, :, 1],
                        Y_test[test_idx:min(test_idx + BATCH_SIZE, test_len)]))
    test_idx += BATCH_SIZE

acc_weight = args.loss_weights[0]
model_save_dir = '%s_%d_split_88_%.1f_%.1f/' % (model_name, patch_length, args.split_ratio, acc_weight)
os.makedirs(model_save_dir, exist_ok=True)
f = open(os.path.join(model_save_dir, 'results.txt'), 'w')
f.write('start time = %s\n' % (time.asctime(time.localtime(time.time()))))
f.write('Train Loss Time Test\n')
f.write('split_ratio = %.3f, loss_weights = %s\n' % (args.split_ratio, args.loss_weights))
f.flush()

acc, avg_acc = test_epoch(model, test_loader)
f.write('normal test acc = %.3f, avg_acc = %.3f\n' % (acc, avg_acc))
f.flush()

actor = agent.attack(model=model, training_data=train_loader, test_data=test_loader,
                     lr=LR, rl_batch=BATCH_SIZE, steps=1, epochs=EPOCH, f=f,
                     patch_len=patch_length, model_save_dir=model_save_dir,
                     loss_weights=args.loss_weights, split_ratio=args.split_ratio)
