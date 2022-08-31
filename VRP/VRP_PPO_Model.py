import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import math
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR
import time
from VRP.vrpUpdate import update_mask, update_state
# from PPORolloutBaselin import RolloutBaseline
from sklearn.preprocessing import MinMaxScaler

INIT = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
max_grad_norm = 2

n_nodes = 21


# device = torch.device('cpu')


class GatConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,
                 negative_slope=0.2, dropout=0):
        super(GatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels + edge_channels, out_channels)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


class Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3, n_heads=4):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(input_node_dim, hidden_node_dim)
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.be = nn.BatchNorm1d(hidden_edge_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)  # 1-16
        # self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_node_dim) for i in range(conv_layers)])
        # self.convs = nn.ModuleList([GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(n_heads)])
        self.convs1 = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])

        # self.convs = nn.ModuleList([GATConv(hidden_node_dim, hidden_node_dim) for i in range(conv_layers)])
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        batch_size = data.num_graphs
        # print(batch_size)
        # edge_attr = data.edge_attr

        x = torch.cat([data.x, data.demand], -1)
        x = self.fc_node(x)
        x = self.bn(x)
        edge_attr = self.fc_edge(data.edge_attr)
        edge_attr = self.be(edge_attr)
        for conv in self.convs1:
            # x = conv(x,data.edge_index)
            x1 = conv(x, data.edge_index, edge_attr)
            x = x + x1

        x = x.reshape((batch_size, -1, self.hidden_node_dim))

        return x


class Attention1(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        super(Attention1, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask):
        '''
        :param state_t: (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        :param context: （batch_size,n_nodes,input_dim）
        :param mask: selected nodes  (batch_size,n_nodes)
        :return:
        '''
        batch_size, n_nodes, input_dim = context.size()
        Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        compatibility = self.norm * torch.matmul(Q, K.transpose(2,
                                                                3))  # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        compatibility = compatibility.squeeze(2)  # (batch_size,n_heads,n_nodes)
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))

        scores = F.softmax(u_i, dim=-1)  # (batch_size,n_heads,n_nodes)
        scores = scores.unsqueeze(2)
        out_put = torch.matmul(scores, V)  # (batch_size,n_heads,1,n_nodes )*(batch_size,n_heads,n_nodes,head_dim)
        out_put = out_put.squeeze(2).view(batch_size, self.hidden_dim)  # （batch_size,n_heads,hidden_dim）
        out_put = self.fc(out_put)

        return out_put  # (batch_size,hidden_dim)


class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = Attention1(n_heads, 1, input_dim, hidden_dim)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask):
        '''
        :param state_t: (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        :param context: （batch_size,n_nodes,input_dim）
        :param mask: selected nodes  (batch_size,n_nodes)
        :return:softmax_score
        '''
        x = self.mhalayer(state_t, context, mask)

        batch_size, n_nodes, input_dim = context.size()
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))  # (batch_size,1,n_nodes)
        compatibility = compatibility.squeeze(1)
        x = torch.tanh(compatibility)
        x = x * (10)
        x = x.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(x, dim=-1)
        return scores


class Decoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder1, self).__init__()

        super(Decoder1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.cell = torch.nn.GRUCell(input_dim*2, hidden_dim*2, bias=True)
        self.prob = ProbAttention(8, input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # self._input = nn.Parameter(torch.Tensor(2 * hidden_dim))
        # self._input.data.uniform_(-1, 1)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, encoder_inputs, pool, actions_old, capcity, demand, n_steps, batch_size, greedy=False,
                _action=False):

        mask1 = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        # 用old_policy来sample的action
        dynamic_capcity = capcity.view(encoder_inputs.size(0), -1)  # bat_size
        demands = demand.view(encoder_inputs.size(0), encoder_inputs.size(1))  # （batch_size,seq_len）
        index = torch.zeros(encoder_inputs.size(0)).to(device).long()
        if _action:
            actions_old = actions_old.reshape(batch_size, -1)
            entropys = []
            old_actions_probs = []

            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]  # depot

                # -----------------------------------------------------------------------------GRU做信息传递
                # hx = self.cell(_input, hx)
                # decoder_input = hx
                # -----------------------------------------------------------------------------pool+cat(first_node,current_node)
                decoder_input = torch.cat([_input, dynamic_capcity], -1)
                decoder_input = self.fc(decoder_input)
                pool = self.fc1(pool)
                decoder_input = decoder_input + pool

                # -----------------------------------------------------------------------------cat(pool,first_node,current_node)
                '''decoder_input = torch.cat([pool, _input, dynamic_capcity], dim=-1)
                decoder_input = self.fc(decoder_input)'''

                if i == 0:
                    mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)

                # decoder_input = torch.cat([pool,_input_first], dim=-1)
                # decoder_input  = self.fc(decoder_input)
                # -----------------------------------------------------------------------------------------------------------
                p = self.prob(decoder_input, encoder_inputs, mask)

                dist = Categorical(p)

                old_actions_prob = dist.log_prob(actions_old[:, i])
                entropy = dist.entropy()
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                old_actions_prob = old_actions_prob * (1. - is_done)
                entropy = entropy * (1. - is_done)

                entropys.append(entropy.unsqueeze(1))
                old_actions_probs.append(old_actions_prob.unsqueeze(1))

                dynamic_capcity = update_state(demands, dynamic_capcity, actions_old[:, i].unsqueeze(-1),
                                               capcity[0].item())
                mask, mask1 = update_mask(demands, dynamic_capcity, actions_old[:, i].unsqueeze(-1), mask1, i)

                _input = torch.gather(encoder_inputs, 1,
                                      actions_old[:, i].unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                                           encoder_inputs.size(
                                                                                               2))).squeeze(1)

            # log_ps = torch.cat(log_ps,dim=1)
            # actions = torch.cat(actions,dim=1)
            entropys = torch.cat(entropys, dim=1)
            old_actions_probs = torch.cat(old_actions_probs, dim=1)
            # log_p = log_ps.sum(dim=1)
            num_e = entropys.ne(0).float().sum(1)
            entropy = entropys.sum(1) / num_e
            old_actions_probs = old_actions_probs.sum(dim=1)

            return 0, 0, entropy, old_actions_probs
        else:
            log_ps = []
            actions = []
            # entropys = []
            # h0 = self.h0.unsqueeze(0).expand(encoder_inputs.size(0), -1)
            # first_cat = self._input[None, :].expand(encoder_inputs.size(0), -1)
            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]  # depot
                # -----------------------------------------------------------------------------GRU做信息传递
                # hx = self.cell(_input, hx)
                # decoder_input = hx
                # -----------------------------------------------------------------------------pool+cat(first_node,current_node)
                decoder_input = torch.cat([_input, dynamic_capcity], -1)
                decoder_input = self.fc(decoder_input)
                pool = self.fc1(pool)
                decoder_input = decoder_input + pool
                # -----------------------------------------------------------------------------cat(pool,first_node,current_node)
                '''decoder_input = torch.cat([pool,_input,dynamic_capcity], dim=-1)
                decoder_input  = self.fc(decoder_input)'''
                # -----------------------------------------------------------------------------------------------------------
                if i == 0:
                    mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)
                p = self.prob(decoder_input, encoder_inputs, mask)
                dist = Categorical(p)
                if greedy:
                    _, index = p.max(dim=-1)
                else:
                    index = dist.sample()

                actions.append(index.data.unsqueeze(1))
                log_p = dist.log_prob(index)
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                log_p = log_p * (1. - is_done)

                log_ps.append(log_p.unsqueeze(1))

                # entropys.append(entropy.unsqueeze(1))
                dynamic_capcity = update_state(demands, dynamic_capcity, index.unsqueeze(-1), capcity[0].item())
                mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)

                _input = torch.gather(encoder_inputs, 1,
                                      index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                               encoder_inputs.size(2))).squeeze(1)
            log_ps = torch.cat(log_ps, dim=1)
            actions = torch.cat(actions, dim=1)

            log_p = log_ps.sum(dim=1)

            return actions, log_p, 0, 0


class Model(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
        super(Model, self).__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.decoder = Decoder1(hidden_node_dim, hidden_node_dim)

    def forward(self, datas, actions_old, n_steps, batch_size, greedy, _action):
        x = self.encoder(datas)  # (batch,seq_len,hidden_node_dim)
        pooled = x.mean(dim=1)
        demand = datas.demand
        capcity = datas.capcity

        actions, log_p, entropy, dists = self.decoder(x, pooled, actions_old, capcity, demand, n_steps, batch_size,
                                                      greedy, _action)
        return actions, log_p, entropy, dists, x


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_node_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Conv1d(hidden_node_dim, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        '''if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)'''

    def forward(self, x):
        x1 = x.transpose(2, 1)
        output = F.relu(self.fc1(x1))
        output = F.relu(self.fc2(output))
        value = self.fc3(output).sum(dim=2).squeeze(-1)
        return value


class Actor_critic(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
        super(Actor_critic, self).__init__()
        self.actor = Model(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.critic = Critic(hidden_node_dim)

    def act(self, datas, actions, steps, batch_size, greedy, _action):
        actions, log_p, _, _, _ = self.actor(datas, actions, steps, batch_size, greedy, _action)

        return actions, log_p

    def evaluate(self, datas, actions, steps, batch_size, greedy, _action):
        _, _, entropy, old_log_p, x = self.actor(datas, actions, steps, batch_size, greedy, _action)

        value = self.critic(x)

        return entropy, old_log_p, value


class Memory:
    def __init__(self):
        self.input_x = []
        # self.input_index = []
        self.input_attr = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.capcity = []
        self.demand = []

    def def_memory(self):
        self.input_x.clear()
        # self.input_index.clear()
        self.input_attr.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.capcity.clear()
        self.demand.clear()


class Agentppo:
    def __init__(self, steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, epoch=1,
                 batch_size=32, conv_laysers=3, entropy_value=0.2, eps_clip=0.2):
        self.policy = Actor_critic(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.old_polic = Actor_critic(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                                      conv_laysers)
        self.old_polic.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps = steps
        self.entropy_value = entropy_value
        self.eps_clip = eps_clip
        self.greedy = greedy
        self._action = True
        self.conv_layers = conv_laysers
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.batch_idx = 1
        self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

    def adv_normalize(self, adv):
        std = adv.std()
        assert std != 0. and not torch.isnan(std), 'Need nonzero std'
        n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
        return n_advs

    def value_loss_gae(self, val_targ, old_vs, value_od, clip_val):
        vs_clipped = old_vs + torch.clamp(old_vs - value_od, -clip_val, +clip_val)
        val_loss_mat_unclipped = self.MseLoss(old_vs, val_targ)
        val_loss_mat_clipped = self.MseLoss(vs_clipped, val_targ)

        val_loss_mat = torch.max(val_loss_mat_unclipped, val_loss_mat_clipped)

        mse = val_loss_mat

        return mse

    def update(self, memory, epoch):
        old_input_x = torch.stack(memory.input_x)
        # old_input_index = torch.stack(memory.input_index)
        old_input_attr = torch.stack(memory.input_attr)
        old_demand = torch.stack(memory.demand)
        old_capcity = torch.stack(memory.capcity)

        old_action = torch.stack(memory.actions)
        old_rewards = torch.stack(memory.rewards).unsqueeze(-1)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(-1)

        datas = []
        edges_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                edges_index.append([i, j])
        edges_index = torch.LongTensor(edges_index)
        edges_index = edges_index.transpose(dim0=0, dim1=1)
        for i in range(old_input_x.size(0)):
            data = Data(
                x=old_input_x[i],
                edge_index=edges_index,
                edge_attr=old_input_attr[i],
                actions=old_action[i],
                rewards=old_rewards[i],
                log_probs=old_log_probs[i],
                demand=old_demand[i],
                capcity=old_capcity[i]
            )
            datas.append(data)
        # print(np.array(datas).shape)
        self.policy.to(device)
        data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=False)
        # 学习率退火
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda f: 0.96 ** epoch)
        value_buffer = 0

        for i in range(self.epoch):

            self.policy.train()
            epoch_start = time.time()
            start = epoch_start
            self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

            for batch_idx, batch in enumerate(data_loader):
                self.batch_idx += 1
                batch = batch.to(device)
                entropy, log_probs, value = self.policy.evaluate(batch, batch.actions, self.steps, self.batch_size,
                                                                 self.greedy, self._action)
                # advangtage function

                # base_reward = self.adv_normalize(base_reward)
                rewar = batch.rewards
                rewar = self.adv_normalize(rewar)
                # rewar = rewar/torch.max(rewar)
                # Value function clipping
                mse_loss = self.MseLoss(rewar, value)

                ratios = torch.exp(log_probs - batch.log_probs)

                # norm advantages
                advantages = rewar - value.detach()

                # advantages = self.adv_normalize(advantages)
                # PPO loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # total loss
                loss = torch.min(surr1, surr2) + 0.5 * mse_loss - self.entropy_value * entropy
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                scheduler.step()

                self.rewards.append(torch.mean(rewar.detach()).item())
                self.losses.append(torch.mean(loss.detach()).item())
                # print(epoch,self.optimizer.param_groups[0]['lr'])

        self.old_polic.load_state_dict(self.policy.state_dict())


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
