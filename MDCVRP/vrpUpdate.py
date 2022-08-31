import torch
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_state(demand,dynamic_capcity,selected,num_depot,c=20):#dynamic_capcity(num,1)

    #Is there a group to access the depot

    current_demand = torch.gather(demand,1,selected)

    dynamic_capcity = dynamic_capcity-current_demand

    for j in range(num_depot):
        depot = selected.squeeze(-1).eq(j)

        if depot.any():
            dynamic_capcity[depot.nonzero().squeeze()] = c

    return dynamic_capcity.detach()#(bach_size,1)


def update_mask(demand,capcity,selected,mask,i,num_depot):
    go_depot = torch.zeros((selected.size(0)),dtype=torch.bool).to(device)
    for j in range(num_depot):
        go_depo = selected.squeeze(-1).eq(j)#If there is a route to select a depot, mask the depot, otherwise it will not mask the depot
        go_depot += go_depo

    #print(go_depot.nonzero().squeeze())
    #visit = selected.ne(0)

    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    if (~go_depot).any():
        for k in range(num_depot):
            mask1[(~go_depot).nonzero(), k] = 0

    if (go_depot).any():
        for k in range(num_depot):
            mask1[(go_depot).nonzero(), k] = 1

    if i+num_depot>demand.size(1):
        is_done = (mask1[:, num_depot:].sum(1) >= (demand.size(1) - num_depot)).float()
        combined = is_done.gt(0)
        mask1[combined.nonzero(), 0] = 0

    a = demand>capcity

    mask = a + mask1

    return mask.detach(),mask1.detach()