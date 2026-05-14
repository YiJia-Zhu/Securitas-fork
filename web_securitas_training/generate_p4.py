import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_size = 32
global_number_choice = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]


parse = argparse.ArgumentParser()
parse.add_argument('--patch-length', type=int, default=800)
parse.add_argument('--policy-dir', default='./DF_800_split_88_0.7_0.6')
parse.add_argument('--max-patch-num', type=int, default=8)
parse.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
parse.add_argument('--gpu', type=int, default=None)
parse.add_argument('--output-dir', default='./patch_p4_code')
args = parse.parse_args()

device = torch.device(args.device)
if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda')

random.seed(2023)
space = [len(global_number_choice) for _ in range(args.patch_length)]


class mind(nn.Module):
    def __init__(self, space):
        super().__init__()

        input_space = copy.deepcopy(space)
        input_space[0] = 10

        self.embedding_list = nn.ModuleList([nn.Embedding(input_space[i], embedding_size) for i in range(len(input_space))])
        self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)
        self.linear_list = nn.ModuleList([nn.Linear(embedding_size, space[i]) for i in range(len(space))])

        self.stage = 0
        self.hidden = None

    def forward(self, x):
        x = self.embedding_list[self.stage](x)
        self.lstm.flatten_parameters()
        x, self.hidden = self.lstm(x, self.hidden)
        prob = self.linear_list[self.stage](x.view(x.size(0), -1))
        return prob

    def increment_stage(self):
        self.stage += 1

    def reset(self):
        self.stage = 0
        self.hidden = None


def select_action(model, state):
    p_a = F.softmax(model(state), dim=1)
    res = torch.argmax(p_a, dim=1)
    return res.unsqueeze(-1), res.unsqueeze(-1)


def select_combo(model):
    state = torch.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(8, 1)).long().to(device)

    combo = []
    log_p_combo = []
    for _ in range(len(space)):
        action, log_p_action = select_action(model, state)
        combo.append(action)
        log_p_combo.append(log_p_action)
        state = action
        model.increment_stage()
    combo = torch.cat(combo, dim=1)
    log_p_combo = torch.cat(log_p_combo, dim=1)
    return combo, log_p_combo


mind_model = mind(space)
mind_model.load_state_dict(torch.load(os.path.join(args.policy_dir, 'best_mind.pth'), map_location='cpu'))
mind_model = mind_model.to(device)
mind_model = mind_model.eval()

combo, _ = select_combo(mind_model)
all_combo = combo.cpu().numpy()
print(all_combo.shape)

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, 'WF_%d.p4' % args.patch_length)

with open(output_path, 'w') as f:
    print('', file=f)
    print('const entries = {', file=f)
    print('// patch_idx, pkt_cnt     mirrored_ip_payload_len (should be 8-unit!!)', file=f)

    combo_idx = [i for i in range(all_combo.shape[0])]
    random.shuffle(combo_idx)

    dummy_ratio = 0.5
    max_patch_num = min(args.max_patch_num, all_combo.shape[0])
    for idx in range(max_patch_num):
        combo = all_combo[combo_idx[idx]]
        for dim1 in range(args.patch_length):
            choice = int(combo[dim1])
            if choice == 0:
                continue

            if dim1 < int(dummy_ratio * args.patch_length):
                print('    (%d,        %d):    ac_should_TTL(%d);' %
                      (idx, dim1 + 1, global_number_choice[choice]), file=f)
            else:
                print('    (%d,        %d):    ac_should_fragment(%d);' %
                      (idx, dim1 + 1, global_number_choice[choice]), file=f)

    print('}', file=f)

print(output_path)
