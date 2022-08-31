import torch
import os
import time
from creat_vrp import reward
from readdata import creat_data
from matplotlib import pyplot as plt
from VRP_Actor import Model
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#from vrp_plt import plot_vehicle_routes
def rollout(model, dataset,n_nodes,Sample,nodes1,T,num_depots):
    # Put in greedy evaluation mode!
    model.eval()
    cost_=[]
    torch_=[]
    def eval_model_bat(bat):
        with torch.no_grad():
            cost1, _ = model(bat, n_nodes * 2,num_depots,Sample,T)
            cost = reward(nodes1, cost1.detach(), n_nodes, num_depots)
        return cost.cpu(),cost1
    for bat in dataset:
        bat.to(device)
        cost,tour1=eval_model_bat(bat)
        cost_.append(cost)
        torch_.append(tour1)
    totall_cost=torch.cat(cost_,0)
    #totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost,tour1

kwargs={'map_location':lambda storage, loc: storage.cuda(0)}
def load_GPU(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(state_dict)
    return model

def evaliuate(path,T=1,Sample=True):
    dl, nodes1,n_nodes,demand,capacity,dl1,num_depots=creat_data(path)
    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
    agent.to(device)
    min_cost = 10000000000
    folder = 'Vrp-52-GAT'
    filename = 'rollout'
    filepath = os.path.join(folder, filename)

    for i in range(84):
        path = os.path.join(filepath, '%s' % i)
        if os.path.exists(path):
            path1 = os.path.join(path, 'actor.pt')
            state_dict = torch.load(path1, map_location='cuda:0')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`[7:]
                new_state_dict[name] = v
            agent.load_state_dict(new_state_dict)

            time_step = time.time()
            cost,tour = rollout(agent, dl, n_nodes,Sample,nodes1,T,num_depots)
            time_step = time.time()-time_step
            if cost<min_cost:
                min_cost=cost
                min_tour=tour.cpu().numpy()
            if Sample:
                print(cost,i, '----Gap', 100 * (cost - 6.1) / 6.1)
            else:
                print(cost,i, 'Sampleing----Gap', '---------', T, 100 * (cost - 6.1) / 6.1)

    print(min_cost)
    return min_cost,time_step
def get_imlist(path):
        path_list = os.listdir(path)
        #path_list.sort(key=lambda x:int(x[:]))
        a = []
        for fikena in path_list:
            a.append(os.path.join(path, fikena))
        return a
filename = './data1'
filename = get_imlist(filename)
results = []
times = []
for path in filename:
    result,time1 = evaliuate(path)
    results.append(result.item())
    times.append(time1)
print(results,times)