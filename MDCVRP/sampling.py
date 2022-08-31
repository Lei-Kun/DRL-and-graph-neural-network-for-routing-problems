import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
import  tqdm
import numpy as np
#from TSPlIbInstance import creat_data,reward
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from creat_vrp import reward1,creat_data,reward,creat_instance
import time
from collections import OrderedDict
from VRP_Actor import Model

from torch.nn import DataParallel
n_nodes = 102
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.data import Data,DataLoader
def rollout(model, dataset,T=1,Sample=False):
    # Put in greedy evaluation mode!
    model.eval()
    def eval_model_bat(bat):
        with torch.no_grad():

            cost, _ = model(bat, n_nodes * 2,Sample,T)

            cost1 = reward1(bat.x,cost.detach(), n_nodes)
        return cost1.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost

def evaliuate(valid_loder,T=1,Sample=False):


    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
    agent.to(device)

    folder = 'Vrp-22-GAT'
    filename = 'rollout'
    filepath = os.path.join(folder, filename)
    path = os.path.join(filepath, '%s' % 69)
    if os.path.exists(path):
        path1 = os.path.join(path, 'actor.pt')
        state_dict = torch.load(path1,map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[:]  # remove `module.`
            new_state_dict[name] = v
        agent.load_state_dict(new_state_dict)

    start=time.time()
    cost = rollout(agent, valid_loder, T,Sample)
    end=time.time()-start

    cost1 = cost.mean()
    cost2=cost.min()
    if Sample:
        print(cost1, 'greedy----Gap', 100 * (cost1 - 6.1) / 6.1)
    else:
        print(cost2, 'Sampleing----Gap','---------',T, 100 * (cost2 - 6.1) / 6.1)

    return cost1,cost2
greedy=0
batch_size=1280
n = []
tours=[]
data_size=10000
for i in range(100):
    datas = []
    node_ = []
    edges_ = []
    demand_=[]
    capcity_=[]
    time1 = time.time()
    for j in range(data_size):
        data, edge, demand, capcity = creat_instance(0,n_nodes)
        node_.append(data)
        edges_.append(edge)
        demand_.append(demand)
        capcity_.append(capcity)
    node_ = np.array(node_)
    edges_ = np.array(edges_)
    demand_ = np.array(demand_)
    capcity_ = np.array(capcity_)
    print(time.time()-time1)
    node_ = np.loadtxt('vrp20_test_data.csv', dtype=np.float, delimiter=',')
    edges_ = np.loadtxt('vrp20_distance.csv', dtype=np.float, delimiter=',')
    demand_=np.loadtxt('vrp20_demand.csv', dtype=np.float, delimiter=',')
    capcity_=np.loadtxt('vrp20_capcity.csv', dtype=np.float, delimiter=',')
    node_,edges_,demand_ = node_.reshape(data_size,n_nodes,2),edges_.reshape(data_size,-1,1),demand_.reshape(-1,n_nodes)

    '''node_, demand_ = node_.reshape(data_size, n_nodes, 2), demand_.reshape(-1, n_nodes)
    # node_,edges_,demand_ = node_.reshape(data_size,n_nodes,2),edges_.reshape(data_size,-1,1),demand_.reshape(-1,n_nodes)
    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5


    edges = np.zeros((data_size, n_nodes, n_nodes, 1))
    for k, data in enumerate(node_):
        for i, (x1, y1) in enumerate(data):
            for j, (x2, y2) in enumerate(data):
                d = c_dist((x1, y1), (x2, y2))
                edges[k][i][j][0] = d
    edges_ = edges.reshape(data_size, -1, 1)'''


    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)
    datas = []

    for i in range(1000):

        data = Data(x=torch.from_numpy(node_[i]).float(), edge_index=edges_index, edge_attr=torch.from_numpy(edges_[i]).float(),
                    demand=torch.tensor(demand_[i]).unsqueeze(-1).float(),
                    capcity=torch.tensor(capcity_[i]).unsqueeze(-1).float())
        datas.append(data)
    # print(datas)
    dl = DataLoader(datas, batch_size=batch_size)

    print('数据创建完毕')

    '''cost,_=evaliuate(dl,1,True)

    if cost>greedy:
        continue'''
    save_node = node_.reshape(-1,2)
    save_edge = edges_.reshape(-1,n_nodes*n_nodes)
    save_demand = demand_.reshape(-1,n_nodes)
    save_capacity=capcity_
    '''np.savetxt('vrp100_test_data.csv', save_node, fmt='%f', delimiter=',')
    np.savetxt('vrp100_distance.csv', save_edge, fmt='%f', delimiter=',')
    np.savetxt('vrp100_demand.csv', save_demand, fmt='%f', delimiter=',')
    np.savetxt('vrp100_capcity.csv', save_capacity, fmt='%f', delimiter=',')'''
    T_cost=[]
#------------------------------------------------------------------------------Sampling
    for T in [2.5]:
        m=[]
        for i in range(1000):
            datas_ = []
            for y in range(1280):
                data = Data(x=torch.from_numpy(node_[i]).float(), edge_index=edges_index,
                            edge_attr=torch.from_numpy(edges_[i]).float(),
                            demand=torch.tensor(demand_[i]).unsqueeze(-1).float(),
                            capcity=torch.tensor(capcity_[i]).unsqueeze(-1).float())

                datas_.append(data)
            dl = DataLoader(datas_, batch_size=batch_size)


            _, sample_cost = evaliuate(dl, T)



            if i%100==0:
                print(sample_cost)


            #tours.append(tour.tolist())
            '''datas1_ = []
            for y in range(12800):
                data = Data(x=torch.from_numpy(node_[i]).float(), edge_index=edges_index,
                            edge_attr=torch.from_numpy(edges_[i]).float())
                datas1_.append(data)
            dl = DataLoader(datas1_, batch_size=batch_size)
            sample_cost3 = sample1(dl, 2.5)
'''
            #cost = evaliuate(dl)
            m.append(sample_cost)
        b = np.mean(np.array(m))
        T_cost.append(b)
        print(T_cost)
        np.savetxt('pn_1', T_cost, fmt='%f', delimiter=',')
    n.append(T_cost)
    print(n)
