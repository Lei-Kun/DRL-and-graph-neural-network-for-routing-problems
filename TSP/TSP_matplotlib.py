import torch
import numpy as np
from TSP.create_tsp_instance import reward1
xyt = torch.tensor([0.1,0.2,0.3])
xyt1 = torch.tensor([1,2,3]).float().unsqueeze(0)
import os
from torch_geometric.data import Data,DataLoader
from TSP.Actor import  Model
#from gurobiBaseline import solve_all_gurobi
from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_tsp(xy, tour, ax1,i,Greedy):
    plt.rc('font', family='Times New Roman', size=22)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color='blue')
    # Starting node
    ax1.scatter([xy[5,0]], [xy[5,1]], s=100, color='red')
    # Arcs
    if Greedy:
        qv = ax1.quiver(
            xs, ys, dx, dy,
            scale_units='xy',
            angles='xy',
            scale=1,
        )
        ax1.set_title('Greedy-{} Nodes, Total Length {:.2f}'.format(len(tour), lengths[-1]), family='Times New Roman')

    else:
        qv = ax1.quiver(
            xs, ys, dx, dy,
            scale_units='xy',
            angles='xy',
            scale=1,
        )

        ax1.set_title('Sampling1280-{} Nodes, Total Length {:.2f}'.format(len(tour), lengths[-1]),
                      family='Times New Roman')
    plt.show()
    #plt.savefig("./temp{}.png".format(i), dpi=600, bbox_inches='tight')
n_nodes=100
def tsp_matplotlib(Greedy=True):
    node_ = np.loadtxt('./test_data/tsp100_test_data.csv', dtype=np.float, delimiter=',')
    node_ = node_.reshape(-1, n_nodes, 2)
    data_size = node_.shape[0]

    x = np.random.randint(1, data_size)

    # Calculate the distance matrix
    edges = np.zeros((n_nodes, n_nodes, 1))

    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5

    for i, (x1, y1) in enumerate(node_[x]):
        for j, (x2, y2) in enumerate(node_[x]):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0] = d
    edges_ = edges.reshape(-1, 1)

    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)
    datas=[]
    for i in range(1):
        # node,edge = creat_instance(n_nodes)
        data = Data(x=torch.from_numpy(node_[x]).float(), edge_index=edges_index,
                    edge_attr=torch.from_numpy(edges_).float())
        datas.append(data)
    #print(datas)

    dl = DataLoader(datas, batch_size=1)

    agent = Model(2, 128, 1, 16, conv_laysers=4).to(device)
    agent.to(device)
    folder = 'trained'
    filepath = os.path.join(folder, '%s' % n_nodes)

    if os.path.exists(filepath):
        path1 = os.path.join(filepath, 'actor.pt')
        agent.load_state_dict(torch.load(path1, device))
    if Greedy:
        print('Data created')
        batch = next(iter(dl))
        batch.to(device)
        agent.eval()
        # -------------------------------------------------------------------------------------------Greedy
        with torch.no_grad():
            tour, _,_ = agent(batch, n_nodes, True)

            #cost = reward1(batch.x, tour.detach(), n_nodes)
            #print(cost)
            #print(tour)
    # -------------------------------------------------------------------------------------------sampling1280
    else:
        datas_ = []
        batch_size1 = 128  # sampling batch_size
        for y in range(1280):

            data = Data(x=torch.from_numpy(node_[x]).float(), edge_index=edges_index,
                        edge_attr=torch.from_numpy(edges_).float())
            datas_.append(data)
        dl = DataLoader(datas_, batch_size=batch_size1)
        print('Data created')
        min_tour = []
        min_cost = 100
        T=1.5
        for batch in dl:
            with torch.no_grad():
                batch.to(device)
                tour1, _ ,_= agent(batch, n_nodes, False, T)
                cost = reward1(batch.x, tour1.detach(), n_nodes)

                id = np.array(cost.cpu()).argmin()
                m_cost = np.array(cost.cpu()).min()
                tour1 = tour1.reshape(batch_size1, -1)
                if m_cost < min_cost:
                    min_cost = m_cost
                    min_tour = tour1[id]

        tour = min_tour.unsqueeze(-2)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tsp(node_[x], tour.squeeze().cpu().numpy(), ax,202,Greedy)


# True:Greedy decoding / False:sampling1280
tsp_matplotlib(Greedy=True)