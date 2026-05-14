import numpy as np
import dpkt
import random
import pickle
import os
import csv

pcaps = []


def dfs(path):
    files= os.listdir(path) # Get all file names under this directory.
    # print(files)

    for file in files: # Traverse this directory.
        if os.path.isdir(path+"/"+file): # Subdirectory.
            dfs(path+"/"+file)
        else: # File.
            # print(path+"/"+file)
            pcaps.append(path+"/"+file)
            # print(pcaps)

def get(ip):
    ip_src = '.'.join(map(str, map(int, ip.src)))
    ip_dst = '.'.join(map(str, map(int, ip.dst)))
    port_src = str(ip.data.sport)
    port_dst = str(ip.data.dport)
    pkt_hash = ip_src + '#' + ip_dst + '#' + port_src + '#' + port_dst
    return pkt_hash

def get_flow(path):
    data_list=[]
    label_list=[]
    total = len(top100)
    for idx, file in enumerate(top100, start=1):
        print('%d/%d %s' % (idx, total, file), flush=True)
        site_path = path + "/" + file
        if os.path.isdir(site_path):
            file2 = os.listdir(site_path + "/")
            for file3 in file2:
                flow = []
                with open(site_path+"/"+file3,'r') as f:
                    reader =csv.reader(f)
                    next(reader)
                    for row in reader:
                        flow.append([int(row[0].split(';')[3]),int(row[0].split(';')[2])])
                while len(flow)<2000:
                    flow.append([0,0])
                flow = flow[:2000]
                # print(flow)
                # print(file)
                data_list.append(flow)
                label_list.append(file)
    return data_list,label_list

if __name__ == '__main__':
    #get_gt()
    print('Start data solve')
    top100 = np.load("./data/tor_100w_2500tr.npz",allow_pickle=True)['labels']
    file_path = "./data/tor_run_v1_001"
    (flow,Label) = get_flow(file_path)
    with open('./data/data_2000.pkl', 'wb') as f:
        pickle.dump(flow, f)
    with open('./data/label_2000.pkl', 'wb') as g:
        pickle.dump(Label, g)
    # print(flow)#1 -1 0



