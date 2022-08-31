
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data,DataLoader

from data_classes import Depot, Customer
def load_problem(path):
    global depots, customers
    capcity = 0
    depots = []
    demands = []
    x_y = []
    customers = []

    with open(path) as f:
        max_vehicles, num_customers, num_depots = tuple(map(lambda z: int(z), f.readline().strip().split()))

        for i in range(num_depots):
            max_duration, max_load = tuple(map(lambda z: int(z), f.readline().strip().split()))

            depots.append(Depot(max_vehicles, max_duration, max_load))
            capcity = max_load


        for i in range(num_customers):
            vals = tuple(map(lambda z: int(z), f.readline().strip().split()))
            cid, x, y, service_duration, demand = (vals[j] for j in range(5))
            customers.append(Customer(cid, x, y, service_duration, demand))
            x_y.append([x,y])
            demands.append(demand)


        for i in range(num_depots):
            vals = tuple(map(lambda z: int(z), f.readline().strip().split()))
            cid, x, y = (vals[j] for j in range(3))
            depots[i].pos = (x, y)
            x_y.append([x,y])
            demands.append(0)

        #demands = demands[0:-num_depots]
        #x_y = x_y[0:-num_depots]

        demands, x_y = np.array(demands), np.array(x_y)
        demands = np.concatenate((demands[-num_depots:],demands[0:-num_depots]))
        x_y = np.concatenate((x_y[-num_depots:], x_y[0:-num_depots]))
        return x_y,demands,np.array([capcity]),num_depots


def creat_instance(path):
    citys,demand,capcity,numdepots = load_problem(path)


    nodes = citys.copy()
    n_nodes=citys.shape[0]

    ys_demand=demand.copy()

    ys_capacity=capcity.copy()

    demand=demand.reshape(-1)

#---------------------------------------------------------------------坐标归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    citys = scaler.fit_transform(citys)

    '''max = np.max(citys)
    citys = citys / (max)'''
#----------------------------------------------------------------------需求归一化
    demand_max = np.max(demand)
    demand = demand / (demand_max)

#----------------------------------------------------------------------容量归一化
    capcity=capcity/demand_max

    def c_dist(x1,x2):
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5
    #edges = torch.zeros(n_nodes,n_nodes)
    edges = np.zeros((n_nodes,n_nodes,1))

    for i, (x1, y1) in enumerate(citys):
        for j, (x2, y2) in enumerate(citys):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0]=d
    edges = edges.reshape(-1, 1)
    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)
    return citys,edges,demand,edges_index,capcity,nodes,n_nodes,ys_demand,ys_capacity,numdepots#demand(num,node) capcity(num)

def creat_data(path,num_samples=1 ,batch_size=1):

    datas = []
    nodes1=[]

    for i in range(num_samples):
        citys,edges,demand,edges_index,capcity,nodes,n_nodes,ys_demand,ys_capacity,numdepots= creat_instance(path)

        data = Data(x=torch.from_numpy(citys).float(), edge_index=edges_index,edge_attr=torch.from_numpy(edges).float(),
                    demand=torch.tensor(demand).unsqueeze(-1).float(),capcity=torch.tensor(capcity).unsqueeze(-1).float())
        datas.append(data)
    datas1 = []
    for i in range(num_samples):
        citys,edges,demand,edges_index,capcity,nodes,n_nodes,ys_demand,ys_capacity,numdepots= creat_instance(path)

        data = Data(x=torch.from_numpy(nodes).float(), edge_index=edges_index,edge_attr=torch.from_numpy(edges).float(),
                    demand=torch.tensor(ys_demand).unsqueeze(-1).float(),capcity=torch.tensor(ys_capacity).unsqueeze(-1).float())
        datas1.append(data)

    #print(datas)
    dl = DataLoader(datas, batch_size=batch_size)
    dl1 = DataLoader(datas1, batch_size=batch_size)
    return dl,nodes,n_nodes,ys_demand,ys_capacity,dl1,numdepots

if __name__ == '__main__':
    path = './data/p01'
    creat_instance(path)
