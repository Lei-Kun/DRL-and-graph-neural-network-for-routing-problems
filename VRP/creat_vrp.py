import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data,DataLoader
from tqdm import tqdm
def creat_instance(num,n_nodes=100,random_seed=None):
    if random_seed is None:
        random_seed = np.random.randint(123456789)
    np.random.seed(random_seed)
    def random_tsp(n_nodes,random_seed=None):

        data = np.random.uniform(0,1,(n_nodes,2))
        return data
    datas = random_tsp(n_nodes)

    def c_dist(x1,x2):
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5
    #edges = torch.zeros(n_nodes,n_nodes)
    edges = np.zeros((n_nodes,n_nodes,1))

    for i, (x1, y1) in enumerate(datas):
        for j, (x2, y2) in enumerate(datas):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0]=d
    edges = edges.reshape(-1, 1)
    CAPACITIES = {
        10: 2.,
        20: 3.,
        50: 4.,
        100: 5.
    }

    demand = np.random.randint(1, 10, size=(n_nodes-1)) # Demand, uniform integer 1 ... 9
    demand = np.array(demand)/10
    demand = np.insert(demand,0,0.)
    capcity = CAPACITIES[n_nodes-1]
    return datas,edges,demand,capcity#demand(num,node) capcity(num)

'''a,s,d,f = creat_instance(2,21)
print(d,f)'''
def creat_data(n_nodes,num_samples=10000 ,batch_size=32):
    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0,dim1=1)

    datas = []

    for i in range(num_samples):
        node, edge, demand, capcity = creat_instance(num_samples, n_nodes)
        data = Data(x=torch.from_numpy(node).float(), edge_index=edges_index,edge_attr=torch.from_numpy(edge).float(),
                    demand=torch.tensor(demand).unsqueeze(-1).float(),capcity=torch.tensor(capcity).unsqueeze(-1).float())
        datas.append(data)
    #print(datas)
    dl = DataLoader(datas, batch_size=batch_size)
    return dl

def reward(static, tour_indices,n_nodes,batch_size):

    static = static.reshape(-1,n_nodes,2)
    #print(static.shape)
    static = static.transpose(2,1)
    tour_indices = tour_indices.reshape(batch_size,-1)
    idx = tour_indices.unsqueeze(1).expand(-1,static.size(1),-1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    #print(tour.shape)
    #print(idx.shape)
    y = torch.cat((tour, tour[:, :1]), dim=1)

    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    #print(tour_len.sum(1))
    return tour_len.sum(1).detach()


def reward1(static, tour_indices,n_nodes):

    static = static.reshape(-1,n_nodes,2)

    static = static.transpose(2,1)

    idx = tour_indices.unsqueeze(1).expand(-1,static.size(1),-1)

    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    #print(tour.shape,tour[0])
    #print(idx.shape,idx[0])
    # Make a full tour by returning to the start
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour,start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    #print(tour_len.sum(1))
    return tour_len.sum(1).detach()