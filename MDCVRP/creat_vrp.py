import numpy as np
import torch
from copy import deepcopy
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
    num_depot = n_nodes%10
    demand = np.random.randint(1, 10, size=(n_nodes-num_depot)) # Demand, uniform integer 1 ... 9
    demand = np.array(demand)/10

    for j in range(num_depot):
        demand = np.insert(demand,0,0.)
    capcity = CAPACITIES[n_nodes-num_depot]
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

def reward(static, tour_indices,n_nodes,num_depotss):

    def c_dist(x1,x2):
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5
    static = static.reshape(-1,n_nodes,2)

    static = torch.from_numpy(static).to('cuda')
    static = static.transpose(2,1)

    tour_indices_1 = deepcopy(tour_indices)

    idx = tour_indices.unsqueeze(1).expand(-1,static.size(1),-1)
    idx_1 = tour_indices_1.unsqueeze(1).expand(-1,static.size(1),-1)


    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    tour_1 = torch.gather(static, 2, idx_1).permute(0, 2, 1)
    start_t = 1000000000
    t_end = 1000000000000
    for i in range(num_depotss):


        start  = c_dist(static.data[:, :, i][0] ,tour[0][0])
        #start = torch.pow(static.data[:, :, i][0],tour[0][0])

        if start_t>start:
            start_t = start
        end = c_dist(static.data[:, :, i][0], tour[0][-1])
        if t_end>end:
            t_end = end
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))
    tour_len = start_t+tour_len.sum(1).unsqueeze(-1).detach()+t_end
    #print(tour.shape,tour[0])
    #print(idx.shape,idx[0])
    # Make a full tour by returning to the start


       #print(tour_len.sum(1))
    return tour_len.detach()
def reward2(static, tour_indices,n_nodes,num_depotss):
    print(num_depotss)
    def c_dist(x1,x2):
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5
    static = static.reshape(-1,n_nodes,2)

    #static = torch.from_numpy(static).to('cuda')
    static = static.transpose(2,1)

    tour_indices_1 = deepcopy(tour_indices)

    idx = tour_indices.unsqueeze(1).expand(-1,static.size(1),-1)
    idx_1 = tour_indices_1.unsqueeze(1).expand(-1,static.size(1),-1)


    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    tour_1 = torch.gather(static, 2, idx_1).permute(0, 2, 1)
    start_t,s_i = 1000000000,0
    t_end,e_i = 1000000000000,0
    for i in range(num_depotss):


        start  = c_dist(static.data[:, :, i][0] ,tour[0][0])
        #start = torch.pow(static.data[:, :, i][0],tour[0][0])

        if start_t>start:
            start_t = start
            s_i = i
        end = c_dist(static.data[:, :, i][0], tour[0][-1])
        if t_end>end:
            t_end = end
            e_i = i
    s_i = torch.tensor([[s_i]]).to('cuda')
    e_i = torch.tensor([[e_i]]).to('cuda')
    tour_indices_1 = torch.cat((s_i,tour_indices_1,e_i),dim=-1)

    print(tour_indices_1)
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))
    tour_len = start_t+tour_len.sum(1).unsqueeze(-1).detach()+t_end
    #print(tour.shape,tour[0])
    #print(idx.shape,idx[0])
    # Make a full tour by returning to the start


       #print(tour_len.sum(1))
    return tour_len.detach()
def reward1(static, tour_indices,n_nodes):

    static = static.reshape(-1,n_nodes,2)
    print(static.shape)
    static = torch.from_numpy(static).to('cuda')
    static = static.transpose(2,1)

    tour_indices_1 = deepcopy(tour_indices)
    for i in range(tour_indices_1.size(0)):
        b = (~tour_indices_1[i].eq(0)).nonzero().reshape(-1).max()
        tour_indices_1[i][b + 1:] = 1


    idx = tour_indices.unsqueeze(1).expand(-1,static.size(1),-1)
    idx_1 = tour_indices_1.unsqueeze(1).expand(-1,static.size(1),-1)


    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    tour_1 = torch.gather(static, 2, idx_1).permute(0, 2, 1)
    #print(tour.shape,tour[0])
    #print(idx.shape,idx[0])
    # Make a full tour by returning to the start
    start = static.data[:, :, 0].unsqueeze(1)
    start_1 = static.data[:,:,1].unsqueeze(1)
    y = torch.cat((start, tour,start), dim=1)
    y_1 = torch.cat((start, tour_1,start_1), dim=1)
    y_2 = torch.cat((start_1, tour, start), dim=1)
    y_3 = torch.cat((start_1, tour_1, start_1), dim=1)
    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    tour_len_1 = torch.sqrt(torch.sum(torch.pow(y_1[:, :-1] - y_1[:, 1:], 2), dim=2))
    tour_len_2 = torch.sqrt(torch.sum(torch.pow(y_2[:, :-1] - y_2[:, 1:], 2), dim=2))
    tour_len_3 = torch.sqrt(torch.sum(torch.pow(y_3[:, :-1] - y_3[:, 1:], 2), dim=2))

    t_len,_ = torch.cat([tour_len.sum(1).unsqueeze(-1).detach(),tour_len_1.sum(1).unsqueeze(-1).detach(),tour_len_2.sum(1).unsqueeze(-1).detach(),tour_len_3.sum(1).unsqueeze(-1).detach()],-1).min(-1)

    #print(tour_len.sum(1))
    return t_len.detach()