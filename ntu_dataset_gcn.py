from dgl.data.utils import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import GraphConv

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
import dgl
import os
import sys


class NTUDataset(DGLDataset):

    def __init__(self, save_graph=True):
        self.save_graph = save_graph
        self.load_skeleton_npy_path = "D:\\maelb\\Documents\\ENSISA\\Programmation\\Python\\Projet 3A\\data\\ntu-rgb\\nturgb+d_skeletons_npy\\"
        self.missing_file_path = "D:\\maelb\\Documents\\ENSISA\\Programmation\\Python\\Projet 3A\\data\\ntu-rgb\\missing_skeletons\\ntu_rgbd_samples_with_missing_skeletons.txt"
        self.save_graphs_dataset_path = "D:\\maelb\\Documents\\ENSISA\\Programmation\\Python\\Projet 3A\\data\\ntu-rgb\\graphs_dataset\\"
        super(NTUDataset, self).__init__(name="ntu")

    def process(self):
        self.graphs = []
        self.labels = []
        self.label_dict = {}
        self._generate()

    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


    def has_cache(self):
        if os.path.exists(self.save_graphs_dataset_path+'dgl_graph_{}.bin'):
            return True

        return False

    def save(self):
        if self.save_graph:
            save_graphs(self.save_graphs_dataset_path+'dgl_graph_{}.bin', self.graphs, {'labels': th.LongTensor(self.labels)})

    def load(self):
        graphs, label_dict = load_graphs(self.save_graphs_dataset_path+'dgl_graph_{}.bin')
        self.graphs = graphs
        self.labels = label_dict['labels']

    @property
    def num_classes(self):
        return 60

    def _generate(self):
        
        step_ranges = list(range(0,100))
        missing_files = self._load_missing_file()
        datalist = os.listdir(self.load_skeleton_npy_path)
        alread_exist = os.listdir(self.save_graphs_dataset_path)
        alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))
        
        graph_id = 0
        for ind, each in enumerate(datalist):
            self._print_toolbar(ind * 1.0 / len(datalist),
                           '({:>5}/{:<5})'.format(
                               ind + 1, len(datalist)
                           ))
            
            S = int(each[1:4])
            if S not in step_ranges:
                continue 
            if each+'.skeleton.npy' in alread_exist_dict:
                print('file already existed !')
                continue
            if each[:20] in missing_files:
                print('file missing')
                continue 
            loadname = self.load_skeleton_npy_path+each
            print(each)
            self.label_dict[graph_id] = int(each[17:20])
            self._build_graph_dgl(loadname, graph_id)
            graph_id+=1
            # raise ValueError()
        self._end_toolbar()
    
    def _build_graph_dgl(self, path, graph_id):
        skeletons = np.load(path, allow_pickle=True).item()
        coordinates = skeletons["skel_body0"][1]
        df = pd.DataFrame(coordinates)
    
        ARTICULATIONS = [x+1 for x in range(25)]
        
        index = 0
        coordinates_articulation = [0] * 25
        pos_articulations = {}
        
        for index in range(len(df)):
            coordinates_articulation[index] = (df[0][index], df[1][index])
        
        for node_number in ARTICULATIONS:
            for coordinates in coordinates_articulation:
                pos_articulations[node_number] = coordinates
                coordinates_articulation.remove(coordinates)
                break
        coordinates_articulation = [0] * 25
        
        src = [0,0,0,1,20,2,16,17,18,12,13,14,20,4,5,6,7,7,20,8,9,10,11,11]
        dst = [16,12,1,20,2,3,17,18,19,13,14,15,4,5,6,7,22,21,8,9,10,11,24,23]
        label = self.label_dict[graph_id]
        g = dgl.graph((th.LongTensor(src), th.LongTensor(dst)), num_nodes=25)
        g = dgl.add_self_loop(g)
        
        attr = []
        for key, value in pos_articulations.items():
            attr.append(value)
        
        g.ndata['position_articulations'] = th.Tensor(attr)
        #data_tuple = (g, th.LongTensor([label]))
        
        self.graphs.append(g)
        self.labels.append(label)
    
    def _print_toolbar(a, rate, annotation=''):
        toolbar_width = 50
        sys.stdout.write("{}[".format(annotation))
        for i in range(toolbar_width):
            if i * 1.0 / toolbar_width > rate:
                sys.stdout.write(' ')
            else:
                sys.stdout.write('-')
            sys.stdout.flush()
        sys.stdout.write(']\r')
    
    def _end_toolbar(a):
        sys.stdout.write('\n')
    
    def _load_missing_file(self):
        missing_files = dict()
        with open(self.missing_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                if line not in missing_files:
                    missing_files[line] = True 
        return missing_files

graphs_dataset_path = "D:\\maelb\\Documents\\ENSISA\\Programmation\\Python\\Projet 3A\\data\\ntu-rgb\\graphs_dataset\\"
dataset = NTUDataset()

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(th.arange(num_train))
test_sampler = SubsetRandomSampler(th.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=1000, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=100, drop_last=False)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
    
# Create the model with given dimensions
model = GCN(len(dataset[0][0].ndata['position_articulations'][0]), 32, dataset.num_classes+1)
optimizer = th.optim.Adam(model.parameters(), lr=0.01)
model.train()

train_losses = []
test_losses = []

for epoch in range(300):
    epoch_loss = 0
    it = 0
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['position_articulations'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        it += 1
    epoch_loss /= (it + 1)
    
    train_losses.append(epoch_loss)
    
    model.eval()
    num_correct = 0
    num_tests = 0
    
    with th.no_grad():
        for batched_graph, labels in test_dataloader:
            pred = model(batched_graph, batched_graph.ndata['position_articulations'].float())
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
        test_losses.append(num_correct/num_tests)
    print('Epoch {} - Training Loss: {}, Test Acc : {}'.format(epoch, epoch_loss, num_correct/num_tests))

plt.figure(figsize=(12,8))

plt.subplot(1,1,1)
plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label='Training Loss', c='blue')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GCN')
plt.legend(loc='upper right', fontsize='x-large')
plt.savefig('gcn_training_loss.png')

plt.subplot(1,1,2)
plt.plot(np.arange(1, len(test_losses) + 1), test_losses, label='Test Loss', c='red')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GCN')
plt.legend(loc='upper right', fontsize='x-large')
plt.savefig('gcn_test_loss.png')

plt.show()