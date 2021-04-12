import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import matplotlib.pyplot as plt
from AirwayGNN import getFileNames,CreateCOOMatrix
from scipy.sparse import coo_matrix
import scipy

def create_airway_graph(bp,conn):
    x = torch.ones(bp, ).float()
    row = torch.tensor(conn.row)
    col = torch.tensor(conn.col)
    edge_index = torch.stack([row, col], dim=0).long()
    data = Data(x=x.reshape(-1, 1), edge_index=edge_index)
    data.y = torch.Tensor([1]).long()
    return data

def create_erdos_renyi_example(n, p=0.25):
    x = torch.ones(n, ).float()
    g = nx.generators.erdos_renyi_graph(n, p)
    conn = nx.convert_matrix.to_scipy_sparse_matrix(g, format='coo')
    row = torch.tensor(conn.row)
    col = torch.tensor(conn.col)
    edge_index = torch.stack([row, col], dim=0).long()
    data = Data(x=x.reshape(-1, 1), edge_index=edge_index)
    return data

def create_regular_example(n, d=8):
    x = torch.ones(n, ).float()
    g = nx.generators.random_regular_graph(d, n)
    conn = nx.convert_matrix.to_scipy_sparse_matrix(g, format='coo')
    row = torch.tensor(conn.row)
    col = torch.tensor(conn.col)
    edge_index = torch.stack([row, col], dim=0).long()
    data = Data(x=x.reshape(-1, 1), edge_index=edge_index)
    return data


class RandomGraphs(InMemoryDataset):

    def __init__(self, root, transform=None):
        super(RandomGraphs, self).__init__(root, transform)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        n_examples = 1024
        n_nodes = 32
        classes = [create_erdos_renyi_example, create_regular_example]
        for i, fn in enumerate(classes):
            for n in range(n_examples):
                data = fn(n_nodes)
                data.y = torch.Tensor([i]).long()
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class AirwayGraphs(InMemoryDataset):
    def __init__(self, root, transform=None):
        super(AirwayGraphs, self).__init__(root, transform)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        filelist = getFileNames(t='xml')
        n_examples = 10
        n_nodes = 20

        for f in filelist[0:n_examples]:
            aw = CreateCOOMatrix(f)
            data = create_airway_graph(aw.shape[0],aw)
            data.y = torch.Tensor([1]).long()
            data_list.append(data)

        for n in range(1,n_examples):
            data = create_erdos_renyi_example(n_nodes)
            data.y = torch.Tensor([0]).long()
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

print("setup data")
dataset2 = AirwayGraphs("AIRWAYGRAPHS").shuffle()

#dataset = RandomGraphs("RANDOMGRAPHS").shuffle()

def run_network(dataset):
    n_examples = len(dataset)
    train_examples = int(0.8 * n_examples)
    valid_examples = int(0.9 * n_examples)
    train_dataset = dataset[:train_examples]
    valid_dataset = dataset[train_examples:valid_examples]
    test_dataset = dataset[valid_examples:]
    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes
    num_tasks = 1
    batch=2 #64 default
    train_loader = DataLoader(train_dataset, batch_size=batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)

    conv_layers = 2
    mlp_layers = 1
    conv_dim = 8
    mlp_dim = 8
    dropout = 0.2


    class AttrProxy(object):
        """Translates index lookups into attribute lookups."""

        def __init__(self, module, prefix):
            self.module = module
            self.prefix = prefix

        def __getitem__(self, i):
            return getattr(self.module, self.prefix + str(i))


    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            input_dim = num_features
            for i in range(conv_layers):
                nn1 = Sequential(Linear(input_dim, conv_dim), ReLU(), Linear(conv_dim, conv_dim))
                self.add_module('conv_' + str(i), GINConv(nn1))
                self.add_module('bn_' + str(i), BatchNorm1d(conv_dim))
                input_dim = conv_dim
            self.conv = AttrProxy(self, 'conv_')
            self.bn = AttrProxy(self, 'bn_')

            for i in range(mlp_layers - 1):
                self.add_module('fc_' + str(i), Linear(input_dim, mlp_dim))
                input_dim = mlp_dim
            self.fc = AttrProxy(self, 'fc_')
            self.fc1 = Linear(input_dim, num_classes)

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            for i in range(conv_layers):
                conv = self.conv.__getitem__(i)
                bn = self.bn.__getitem__(i)
                x = F.relu(conv(x, edge_index))
                x = bn(x)
            x = global_add_pool(x, batch)
            for i in range(mlp_layers - 1):
                fc = self.fc.__getitem__(i)
                x = F.relu(fc(x))
                x = F.dropout(x, p=dropout, training=self.training)
            x = self.fc1(x)
            return x


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


    def train(epoch):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()


    def test(loader):
        model.eval()
        targets = []
        preds = []
        for data in loader:
            data = data.to(device)
            output = model(data)
            preds.append(output.max(dim=1)[1].cpu().numpy())
            targets.append(data.y.cpu().numpy())

        targets = np.concatenate(targets, axis=0)
        preds = np.concatenate(preds, axis=0).reshape((-1, num_tasks))
        acc_scores = accuracy_score(targets, preds)
        return acc_scores


    best_val_acc = 0.
    for epoch in range(1, 101):
        train(epoch)
        val_acc = test(valid_loader)
        if val_acc >= best_val_acc or epoch == 1:
            best_val_acc = val_acc
            test_acc = test(test_loader)
            # torch.save(model.state_dict(), 'params.pt')
        print('Epoch: {:03d}, Valid Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch,val_acc, test_acc))

#run_network(dataset2)

# n_examples = len(dataset)
# print(n_examples)
# num_features = dataset.num_features
# num_classes = dataset.num_classes
# print(num_classes,num_features)

n_examples = len(dataset2)
print(n_examples)
num_features = dataset2.num_features
num_classes = dataset2.num_classes
print(num_classes,num_features)