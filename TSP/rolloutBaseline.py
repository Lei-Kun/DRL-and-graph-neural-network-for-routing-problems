import torch

from scipy.stats import ttest_rel
import copy

from TSP.create_tsp_instance import reward1

from torch.nn import DataParallel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def rollout(model, dataset, n_nodes):
    model.eval()
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ ,_= model(bat,n_nodes,True)
            cost = reward1(bat.x,cost.detach(), n_nodes)
        return cost.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost

class RolloutBaseline():

    def __init__(self, model,  dataset, n_nodes=50,epoch=0):
        super(RolloutBaseline, self).__init__()
        self.n_nodes = n_nodes
        self.dataset = dataset
        self._update_model(model, epoch)
    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        self.bl_vals = rollout(self.model, self.dataset, n_nodes=self.n_nodes).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, x, n_nodes):
        with torch.no_grad():
            tour, _ ,_= self.model(x,n_nodes,True)
            v= reward1(x.x, tour.detach(), n_nodes)
        return v

    def epoch_callback(self, model, epoch):

        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.n_nodes).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {} , baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            # Calc p value
            print('[[[[[[[[[[[[[[[[[[',candidate_vals, self.bl_vals)
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            print('[[[[[[[[[[[[[[[[[[',t,p)
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])