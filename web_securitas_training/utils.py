import torch
import torch.nn as nn
import numpy as np

random_seed = 16
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

loss_func = nn.CrossEntropyLoss()

def test_epoch(model, test_data):
    corrects = 0
    avg_loss = 0
    test_size = 0
    
    global_acc = [0 for _ in range(100)]
    class_num = [0 for _ in range(100)]
    
    for _, (b_x, b_x_size, b_y) in enumerate(test_data):
        test_size += b_x.shape[0]
        b_x = torch.from_numpy(b_x).cuda() # float64 type
        b_y = torch.from_numpy(b_y).cuda()
        logit = model(b_x.float())[0]
        loss = loss_func(logit, b_y)
        avg_loss += loss.item()
        
        pred = torch.max(logit, 1)[1].view(b_y.size()).cpu().numpy()
        gt = b_y.cpu().numpy()
        for idx in range(pred.shape[0]):
            class_num[gt[idx]] += 1
            if pred[idx] == gt[idx]:
                global_acc[pred[idx]] += 1
                
        #corrects += (torch.max(logit, 1)[1].view(b_y.size()).data == b_y.data).sum()

    #accuracy = 100.0 * corrects / test_size
    acc = sum(global_acc) / sum(class_num)
    temp = [0 for _ in range(100)]
    for idx in range(100):
        temp[idx] = global_acc[idx] / class_num[idx]
    avg_acc = sum(temp) / 100

    return acc * 100, avg_acc * 100