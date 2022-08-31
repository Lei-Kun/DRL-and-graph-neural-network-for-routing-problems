import numpy as np
import torch
from torch_geometric.data import Data,DataLoader

def creat_instance(n_nodes=100,random_seed=None):
    def random_tsp(n_nodes,random_seed=random_seed):
        if random_seed is None:
            random_seed = np.random.randint(123456789)
        np.random.seed(random_seed)

        data = np.random.uniform(0,1,(n_nodes,2))
        return data
    data = random_tsp(n_nodes)#Generate a set of tsp instances

    #Calculate the distance matrix
    def c_dist(x1,x2):
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5

    edges = np.zeros((n_nodes,n_nodes,1))
    for i, (x1, y1) in enumerate(data):
        for j, (x2, y2) in enumerate(data):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0]=d

    edges = edges.reshape(-1,1)
    return data,edges

def creat_data(n_nodes=20,num_samples=10000 ,batch_size=32):
    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0,dim1=1)

    datas = []

    for i in range(num_samples):
        node,edge = creat_instance(n_nodes)
        data = Data(x=torch.from_numpy(node).float(), edge_index=edges_index,edge_attr=torch.from_numpy(edge).float())
        datas.append(data)
    #print(datas)
    dl = DataLoader(datas, batch_size=batch_size)
    return dl

#nodes(batch_size,2,n_nodes)
#edges(batch_size,n_nodes,n_nodes,1)
#edges_index(batch_size,2,n_nodes)
#dynamic(batch_size,1,n_nodes)
def reward(static, tour_indices,n_nodes,batch_size):

    static = static.reshape(-1,n_nodes,2)
    #print(static.shape)
    static = static.transpose(2,1)
    tour_indices = tour_indices.reshape(batch_size,n_nodes)
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    #print(tour.shape)
    #print(idx.shape)
    y = torch.cat((tour, tour[:, :1]), dim=1)

    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    #print(tour_len.sum(1))
    return tour_len.sum(1).detach()


def reward1(static, tour_indices,n_nodes):
    #static = static.transpose(2,1)
    #print(static.shape)  static(batch_size*n_nodes,2)
    #print(static.shape,static)
    static = static.reshape(-1,n_nodes,2)
    #print(static.shape,static)
    static = static.transpose(2,1)
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    #print(tour.shape,tour[0])
    #print(idx.shape,idx[0])
    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    #print(tour_len.sum(1))
    return tour_len.sum(1).detach()










