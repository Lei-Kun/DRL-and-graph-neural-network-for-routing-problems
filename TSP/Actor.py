import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import math
from torch_geometric.nn import GATConv
from torch.distributions.categorical import Categorical


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_nodes = 50

# device = torch.device('cpu')
#E-GAT NETWORK
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

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out

#GAT NETWORK
class GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(GATConv, self).__init__(aggr='add')
        self.negative_slope = negative_slope
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index_i, size_i):
        # x_j -> x_i
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
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
        # self.layer_n = nn.LayerNorm(hidden_node_dim, eps=1e-6)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)  # 1-16
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(hidden_node_dim) for i in range(conv_layers)])
        self.convs = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])


    def forward(self, data):
        batch_size = data.num_graphs
        # print(batch_size)
        # edge_attr = data.edge_attr

        x = self.fc_node(data.x)

        x = self.bn(x)
        edge_attr = self.fc_edge(data.edge_attr)
        edge_attr = self.be(edge_attr)

        for (conv, bn) in zip(self.convs, self.bn1):
            # x = conv(x,data.edge_index)
            x1 = conv(x, data.edge_index, edge_attr)
            # x1 = conv1(x1, data.edge_index, edge_attr)# if Build encoder with GAT, no edge
            x = x + x1 # Residual connection
        x = x.reshape((batch_size, -1, self.hidden_node_dim))
        return x


#multi-head attention ----- first layer of decoder
class MHAlayer(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        super(MHAlayer, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.dropout = nn.Dropout(attn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm = 1 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

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

#second layer of decoder
class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)

        self.mhalayer = MHAlayer(n_heads, 1, input_dim, hidden_dim)

    def forward(self, state_t, residual, context, mask,T):
        '''
        :param state_t: (batch_size,1,input_dim(GATembeding,fist_node,end_node))
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

        scores = F.softmax(x/T, dim=-1)
        return scores


class Decoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.prob = ProbAttention(8, input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.cell = torch.nn.GRUCell(input_dim, hidden_dim, bias=True)
        self._input = nn.Parameter(torch.Tensor(2 * hidden_dim))
        self._input.data.uniform_(-1, 1)

    def forward(self, encoder_inputs, pool, n_steps, T,greedy=False):
        _input = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(2))).to(device)
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1))).to(device)

        log_ps = []
        actions = []
        first_cat = self._input[None, :].expand(encoder_inputs.size(0), -1)
        for i in range(n_steps):
            if i == 0:
                _input_first = first_cat

            else:
                _input_first = torch.cat([first_node, _input], dim=-1)

            # -----------------------------------------------------------------------------pool+cat(first_node,current_node)
            _input_first = self.fc(_input_first)
            pool = self.fc1(pool)
            decoder_input = _input_first + pool
            # -----------------------------------------------------------------------------cat(pool,first_node,current_node)

            # decoder_input = torch.cat([pool,_input_first], dim=-1)
            # decoder_input  = self.fc(decoder_input)
            # -----------------------------------------------------------------------------------------------------------
            #                 print (hx.size(),encoder_inputs.size(),mask.size())
            p = self.prob(decoder_input, decoder_input, encoder_inputs, mask,T)

            dist = Categorical(p)
            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()
            actions.append(index.data.unsqueeze(1))
            log_p = dist.log_prob(index)
            log_ps.append(log_p.unsqueeze(1))

            mask = mask.scatter(1, index.unsqueeze(-1).expand(mask.size(0), -1), 1)

            _input = torch.gather(encoder_inputs, 1,
                                  index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                           encoder_inputs.size(2))).squeeze(1)
            if i == 0:
                first_node = _input
        log_ps = torch.cat(log_ps, dim=1)
        actions = torch.cat(actions, dim=1)
        log_p = log_ps.sum(dim=1)
        return actions, log_p


class Model(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers=3):
        super(Model, self).__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                               conv_layers=conv_laysers)
        self.decoder = Decoder1(hidden_node_dim, hidden_node_dim)

    def forward(self, datas, steps, greedy=False,T=1):
        x = self.encoder(datas)  # (batch,seq_len,hidden_node_dim)
        pooled = x.mean(dim=1)
        actions, log_p = self.decoder(x, pooled, steps,T,greedy)

        return actions, log_p, pooled





if __name__ == '__main__':
    raise Exception('Cannot be called from main')
