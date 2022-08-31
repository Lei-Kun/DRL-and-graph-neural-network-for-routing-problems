
import os
import numpy as np
import torch
from creat_vrp import reward2
from collections import OrderedDict
from torch_geometric.data import Data,DataLoader
from VRP_Actor import Model
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_nodes = 102

def discrete_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(cost,data, route, ax1, Greedy, markersize=5, visualize_demands=False, demand_scale=1,
                            round_demand=False):

        plt.rc('font', family='Times New Roman', size=10)
        for i,j in enumerate(route):
            if j==1:
                route[i]=0
        routes = [r[r != 0] for r in np.split(route.cpu().numpy(), np.where(route.cpu().numpy() == 0)[0]) if    (r != 0).any()]
        print(routes)
        routes =[np.array([  0,  91,   9,  85,  83,  21,  89,  82,  16,  90,  15,  67,  47,  49,
           7,1]),   np.array([1,  84,  35,  33,  34,  17,  94,  96,  42,   1]),np.array([1,  52,  60,  69,
          95,   6,  11,  87,   2,  66,   0]),np.array([0,  79,  98,  38,  27,  64,  40,  14,
          37,  20, 101,   1]),np.array([1,  53,  72,  57,   8,  59,  43,  54,  80,   1]),np.array([1,  24,
           3,  18,  77,  74,  58,   1]),np.array([1,  13, 100,   4,  28,  97,  88,  86,  55,
          76,   0]),np.array([0,  93,  51,  44,  48,  12,  70,  63,  39,   0]),np.array([0,  30,  68,  78,
          61,  46,  22,  71,   0]),np.array([0,  29,  99,  73,   0]),np.array([0,  92,  50,  26,  45,  56,
          41,  75,  65,  62,   0]),np.array([0,  32,   5,  31,  23,  81,  25,  10,  36,  19,
           0])]
        '''routes = [np.array([  0,  91,   9,  85,  21,  83,  89,  82,  16,  90,  15,  67,  47,  49,
           7,   1]),np.array([1,  84,  35,  33,  34,  17,  94,  42,   1]),np.array([1,  52,  60,  69,  95,
           6,  11,  87,   2,  66,   0]),np.array([0,  79,  98,  27,  38,  64,  40,  14,  72,
           1]),np.array([1, 101,  96,   3,  18,  77,  74,  58,  24,   1]),np.array([1,  53,  37,  57,  20,
          80,  43,  54,  59,   8,  55,   0]),np.array([0,  48,  51,  93,  44,  76,  12,  19,
           0]),np.array([0,  29,  99,  73,   0]),np.array([0,  88,  86,  63,  70,  50,  71,  92,  41,  65,
          39,   0]),np.array([0,   4, 100,  13,  28,  97,  68,  78,  30,   0]),np.array([0,  45,  61,  46,
          22,  26,  56,  75,  62,   0]),np.array([0,  36,  10,  32,   5,  31,  23,  81,  25,
           0])]'''


        depot = data.x[0].cpu().numpy()
        depot1 = data.x[1:2].cpu().numpy()
        locs = data.x[:].cpu().numpy()
        demands = data.demand.cpu().numpy() * 10
        demands = demands[:]

        capacity = data.capcity * 10

        x_dep, y_dep = depot
        x1_d, y1_d = depot1[0]
        ax1.plot(x_dep, y_dep, 'sk', markersize=markersize * 4)
        ax1.plot(x1_d, y1_d, 'sk', markersize=markersize * 4)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        #plt.show()
        legend = ax1.legend(loc='upper center')

        cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
        dem_rects = []
        used_rects = []
        cap_rects = []
        qvs = []
        total_dist = 0
        shape = ['o','v','^','D','<','>','s','p','.','h','H','d']
        for veh_number, r in enumerate(routes):
            #color = cmap(len(routes) - veh_number)  # Invert to have in rainbow order
            color = 'k'
            shap = shape[veh_number]
            route_demands = demands[r ]
            coords = locs[r , :]
            xs, ys = coords.transpose()

            total_route_demand = sum(route_demands)
            # assert total_route_demand <= capacity

            dist = 0
            x_prev, y_prev = x_dep, y_dep
            cum_demand = 0
            for (x, y), d in zip(coords, route_demands):
                dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

                cap_rects.append(Rectangle((x, y), 0.01, 0.1))
                used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
                dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))

                x_prev, y_prev = x, y
                cum_demand += d
            widths = np.linspace(0, 2)
            dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
            total_dist += dist
            if not visualize_demands:
                ax1.plot(xs, ys, shap, mfc=color, markersize=markersize*2, markeredgewidth=0.0,label='R{}, N({}), C {} / {}, D {:.2f}'.format(veh_number,
                    len(r),int(total_route_demand) if round_demand else total_route_demand,
                    int(capacity) if round_demand else capacity, dist))

            qv = ax1.quiver(
                xs[:-1],
                ys[:-1],
                xs[1:] - xs[:-1],
                ys[1:] - ys[:-1],
                scale_units='xy',
                width=0.0035,
                angles='xy',
                scale=1,
                #color=color,

            )

            qvs.append(qv)
        if Greedy:
            ax1.set_title('Greedy;总路线数:{};总距离:{:.2f}'.format(len(routes), cost.item()),
                          family='SimSun', size=20)
        else:
            ax1.set_title('OR-tools;总路线数:{};总距离:{:.2f}'.format(len(routes), 14.77),
                          family='SimSun', size=20)

        ax1.legend(handles=qvs)
        plt.legend(loc=1)
        pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
        pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
        pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')

        if visualize_demands:
            ax1.add_collection(pc_cap)
            ax1.add_collection(pc_used)
            ax1.add_collection(pc_dem)
        #plt.show()

        plt.savefig("./100-2-2{}.svg".format(1), dpi=300, bbox_inches='tight')

def vrp_matplotlib(Greedy=True):

    node_ = np.loadtxt('vrp100_test_data.csv', dtype=np.float, delimiter=',')
    demand_=np.loadtxt('vrp100_demand.csv', dtype=np.float, delimiter=',')
    capcity_=np.loadtxt('vrp100_capcity.csv', dtype=np.float, delimiter=',')

    node_,demand_=node_.reshape(-1,n_nodes,2),demand_.reshape(-1,n_nodes)
    data_size = node_.shape[0]

    #x=np.random.randint(1,data_size)
    x = 25
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

    datas = []
    data = Data(x=torch.from_numpy(node_[x]).float(), edge_index=edges_index, edge_attr=torch.from_numpy(edges_).float(),
                demand=torch.tensor(demand_[x]).unsqueeze(-1).float(),
                capcity=torch.tensor(capcity_[x]).unsqueeze(-1).float())
    datas.append(data)

    data_loder = DataLoader(datas, batch_size=1)



    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
    agent.to(device)
    folder = 'Vrp-102-GAT'
    filename = 'rollout'
    filepath = os.path.join(folder, filename)
    path = os.path.join(filepath, '%s' % 51)
    if os.path.exists(path):
        path1 = os.path.join(path, 'actor.pt')
        state_dict = torch.load(path1, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v


        agent.load_state_dict(new_state_dict)
    if Greedy:
        batch = next(iter(data_loder))
        batch.to(device)
        agent.eval()
        #-------------------------------------------------------------------------------------------Greedy
        with torch.no_grad():
            tour, _ = agent(batch, n_nodes * 2,2,True)
            cost = reward2(batch.x, tour.detach(), n_nodes, 2)

            #print(cost)
            #print(tour)
    #-------------------------------------------------------------------------------------------sampling1280
    else:
        datas_ = []
        batch_size1 = 1  # sampling batch_size
        for y in range(1):
            data = Data(x=torch.from_numpy(node_[x]).float(), edge_index=edges_index,
                        edge_attr=torch.from_numpy(edges_).float(),
                        demand=torch.tensor(demand_[x]).unsqueeze(-1).float(),
                        capcity=torch.tensor(capcity_[x]).unsqueeze(-1).float())
            datas_.append(data)
        dl = DataLoader(datas_, batch_size=batch_size1)

        min_tour=[]
        min_cost=100
        T=1.2#Temperature hyperparameters
        for batch in dl:
            with torch.no_grad():
                batch.to(device)
                tour1, _ = agent(batch, n_nodes * 2,2,False, T)
                cost = reward2(batch.x, tour1.detach(), n_nodes,2)

                id = np.array(cost.cpu()).argmin()
                m_cost=np.array(cost.cpu()).min()
                tour1=tour1.reshape(batch_size1,-1)
                if m_cost<min_cost:
                    min_cost=m_cost
                    min_tour=tour1[id]

        tour=min_tour.unsqueeze(-2)

    #--------------------------------------------------------------------------------------------
    for i, (data, tour) in enumerate(zip(data_loder, tour)):
        if Greedy:
            print(data.x,data.demand,tour)
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_vehicle_routes(cost,data, tour, ax,Greedy, visualize_demands=False, demand_scale=50, round_demand=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_vehicle_routes(cost,data, tour, ax,Greedy, visualize_demands=False, demand_scale=50, round_demand=True)

#True:Greedy decoding / False:sampling1280
vrp_matplotlib(Greedy=True)