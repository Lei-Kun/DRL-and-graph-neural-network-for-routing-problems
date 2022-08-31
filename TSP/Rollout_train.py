import datetime
import numpy as np
import torch
import os
import time
import torch.nn as nn

import torch.optim as optim
from TSP.Actor import Model
from TSP.create_tsp_instance import creat_data,reward1
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from torch.optim.lr_scheduler import LambdaLR
from TSP.rolloutBaseline import RolloutBaseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
n_nodes = 50 # Problem size
def rollout(model, dataset,batch_size, n_nodes):
    model.eval()
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ ,_= model(bat,n_nodes,True)
            cost = reward1(bat.x,cost.detach(), n_nodes)
        return cost.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost
def initWeights(net, scheme='orthogonal'):

   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            nn.init.orthogonal_(e)
      elif scheme == 'normal':
         nn.init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         nn.init.xavier_normal(e)
max_grad_norm = 2

rewardss = []
def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

def train():
    #------------------------------------------------------------------------------------------------------------------------------
    class RunBuilder():
        @staticmethod  # 不用产生实例对象就可以直接调用该类
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    params = OrderedDict(
        lr=[1e-4],
        batch_size=[128],
        hidden_node_dim=[128],
        hidden_edge_dim=[16],
        conv_laysers=[4],
        data_size=[12800]
    )
    runs = RunBuilder.get_runs(params)#一次训练多个超参数
    #-------------------------------------------------------------------------------------------------------------------------------------

    folder = 'Tsp-{}-GAT'.format(n_nodes)
    filename = 'rollout'
    for lr,batch_size,hidden_node_dim,hidden_edge_dim,conv_laysers,data_size in runs:
        print('lr','batch_size','hidden_node_dim','hidden_edge_dim','conv_laysers:',lr,batch_size,hidden_node_dim,hidden_edge_dim,conv_laysers)
        data_loder = creat_data(n_nodes, data_size,batch_size=batch_size)#Training set
        valid_loder = creat_data(n_nodes, 1280, batch_size=batch_size)#Validation set

        print('Data creation completed')

        actor = Model(2, hidden_node_dim, 1, hidden_edge_dim, conv_laysers=conv_laysers).to(device)
        rol_baseline = RolloutBaseline(actor,valid_loder,n_nodes=n_nodes)
        #initWeights(actor)


        filepath = os.path.join(folder, filename)
        '''path = os.path.join(filepath,'%s' % 3)
                if os.path.exists(path):
                    path1 = os.path.join(path, 'actor.pt')
                    self.agent.old_polic.load_state_dict(torch.load(path1, device))'''


        now = '%s' % datetime.datetime.now().time()
        now = now.replace(':', '_')
        actor_optim = optim.Adam(actor.parameters(), lr=lr)
        costs = []
        for epoch in range(100):

            print("epoch:",epoch,"------------------------------------------------")
            actor.train()

            times, losses, rewards, critic_rewards = [], [], [], []
            epoch_start = time.time()
            start = epoch_start

            scheduler = LambdaLR(actor_optim, lr_lambda=lambda f: 0.96 ** epoch)
            for batch_idx, batch in enumerate(data_loder):
                batch = batch.to(device)
                tour_indices, tour_logp,_ = actor(batch,n_nodes)
                reward = reward1(batch.x, tour_indices.detach(),n_nodes)
                #rewar = adv_normalize(reward)

                base_reward = rol_baseline.eval(batch,n_nodes)
                #base_reward = adv_normalize(base_reward)
                advantage = (reward - base_reward)
                print('reward', reward.mean(), base_reward.mean())
                print('advantage', torch.mean(advantage))
                print('log_p', torch.mean(tour_logp).item())
                #advantage = adv_normalize(advantage)
                actor_loss = torch.mean(advantage.detach() * tour_logp)

                actor_optim.zero_grad()
                actor_loss.backward()
                #grad_norms = clip_grad_norms(actor_optim.param_groups, 1)
                #torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()
                scheduler.step()

                rewards.append(torch.mean(reward.detach()).item())
                losses.append(torch.mean(actor_loss.detach()).item())

                step = 10
                if (batch_idx + 1) % step == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end

                    mean_loss = np.mean(losses[-step:])
                    mean_reward = np.mean(rewards[-step:])

                    print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                          (batch_idx, len(data_loder), mean_reward, mean_loss,
                           times[-1]))

            rol_baseline.epoch_callback(actor,epoch)
            epoch_dir = os.path.join(filepath, '%s' % epoch)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            cost = rollout(actor, valid_loder, batch_size, n_nodes)
            cost = cost.mean()
            costs.append(cost.item())
            print('Problem:TSP''%s' % n_nodes, '/ Average distance:', cost.item())
            print(costs)

train()