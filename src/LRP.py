import re
import torch
from torch.functional import norm
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math, os


class ClustDistLayer(nn.Module):
    def __init__(self, centroids, n_clusters, clust_list, device):
        super(ClustDistLayer, self).__init__()
        self.centroids = Variable(centroids).to(device)
        self.n_clusters = n_clusters
        self.clust_list = clust_list

    def forward(self, x, curr_clust_id):
        output = []
        for i in self.clust_list:
            if i==curr_clust_id:
                continue
            weight = 2 * (self.centroids[self.clust_list.index(curr_clust_id)] - self.centroids[self.clust_list.index(i)])
            bias = torch.norm(self.centroids[self.clust_list.index(curr_clust_id)], p=2) - torch.norm(self.centroids[self.clust_list.index(i)], p=2)
            h = torch.matmul(x, weight.T) + bias
            output.append(h.unsqueeze(1))

        return torch.cat(output, dim=1)


class ClustMinPoolLayer(nn.Module):
    def __init__(self, beta):
        super(ClustMinPoolLayer, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def forward(self, inputs):
        return - torch.log(torch.sum(torch.exp(inputs * -self.beta), dim=1) + self.eps)


class LRP(nn.Module):
    def __init__(self, model, X1, X2, Z, clust_ids, n_clusters, beta=1., device="cuda"):
        super(LRP, self).__init__()
        #model.freeze_model()
        self.model = model
        self.clust_ids = clust_ids
        self.n_clusters = n_clusters
        self.clust_list = np.unique(clust_ids).astype(int).tolist()
        self.centroids_ =torch.tensor(self.set_centroids(Z), dtype=torch.float32)
        self.X1_ = torch.tensor(X1, dtype=torch.float32)
        self.X2_ = torch.tensor(X2, dtype=torch.float32)
        self.Z_ = torch.tensor(Z, dtype=torch.float32)
        self.distLayer = ClustDistLayer(self.centroids_, n_clusters, self.clust_list, device).to(device)
        self.clustMinPool = ClustMinPoolLayer(beta).to(device)
        self.device = device

    def set_centroids(self, Z):
        centroids = []
        for i in self.clust_list:
            clust_Z = Z[self.clust_ids==i]
            curr_centroid = np.mean(clust_Z, axis=0)
            centroids.append(curr_centroid)

        return np.stack(centroids, axis=0)

    def clust_minpoolAct(self, X1, X2, curr_clust_id):
        z,_,_,_,_,_,_,_,_ = self.model.forwardAE(X1, X2)
        return self.clustMinPool(self.distLayer(z, curr_clust_id))

    def calc_carlini_wagner_one_vs_one(self, clust_c_id, clust_k_id, margin=1., lamda=1e2, max_iter=5000, lr=2e-3, use_abs=True):
        X1_0 = Variable(self.X1_[self.clust_ids==clust_c_id], requires_grad=False).to(self.device)
        curr_X1 = Variable(X1_0 + 1e-6, requires_grad=True).to(self.device)
        X2_0 = Variable(self.X2_[self.clust_ids==clust_c_id], requires_grad=False).to(self.device)
        curr_X2 = Variable(X2_0 + 1e-6, requires_grad=True).to(self.device)
        optimizer = optim.SGD([curr_X1, curr_X2], lr=lr)

        for iter in range(max_iter):
            clust_c_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_c_id)
            clust_k_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_k_id)
            clust_loss_tensor = margin + clust_c_minpoolAct_tensor - clust_k_minpoolAct_tensor
            clust_loss_tensor = torch.maximum(clust_loss_tensor, torch.zeros_like(clust_loss_tensor))
            clust_loss = torch.sum(clust_loss_tensor)

            norm_loss = torch.norm(curr_X1 - X1_0, p=1) + torch.norm(curr_X2 - X2_0, p=1)

            loss = clust_loss * lamda + norm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter+1) % 50 == 0:
                print('Iteration {}, Total loss:{:.8f}, clust loss:{:.8f}, L1 penalty:{:.8f}'.format(iter, loss.item(), clust_loss.item(), norm_loss.item()))

        if use_abs:
            rel_score1 = torch.mean(torch.abs(curr_X1 - X1_0), dim=0)
            rel_score2 = torch.mean(torch.abs(curr_X2 - X2_0), dim=0)
        else:
            rel_score1 = torch.mean(curr_X1 - X1_0, dim=0)
            rel_score2 = torch.mean(curr_X2 - X2_0, dim=0)
        return rel_score1.data.cpu().numpy(), rel_score2.data.cpu().numpy()

    def calc_carlini_wagner_one_vs_rest(self, clust_c_id, margin=1., lamda=1e2, max_iter=5000, lr=2e-3, use_abs=True):
        X1_0 = Variable(self.X1_[self.clust_ids==clust_c_id], requires_grad=False).to(self.device)
        curr_X1 = Variable(X1_0 + 1e-6, requires_grad=True).to(self.device)
        X2_0 = Variable(self.X2_[self.clust_ids==clust_c_id], requires_grad=False).to(self.device)
        curr_X2 = Variable(X2_0 + 1e-6, requires_grad=True).to(self.device)
        optimizer = optim.SGD([curr_X1, curr_X2], lr=lr)

        for iter in range(max_iter):
            clust_rest_minpoolAct_tensor_list = []
            for clust_k_id in self.clust_list:
                if clust_k_id == clust_c_id:
                    continue
                clust_k_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_k_id)
                clust_rest_minpoolAct_tensor_list.append(clust_k_minpoolAct_tensor)
            clust_rest_minpoolAct_tensor = clust_rest_minpoolAct_tensor_list[0]
            for clust_k_id in range(1, len(clust_rest_minpoolAct_tensor_list)):
                clust_rest_minpoolAct_tensor = torch.maximum(clust_rest_minpoolAct_tensor, clust_rest_minpoolAct_tensor_list[clust_k_id])

            clust_c_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_c_id)

            clust_loss_tensor = margin + clust_c_minpoolAct_tensor - clust_rest_minpoolAct_tensor
            clust_loss_tensor = torch.maximum(clust_loss_tensor, torch.zeros_like(clust_loss_tensor))
            clust_loss = torch.sum(clust_loss_tensor)

            norm_loss = torch.norm(curr_X1 - X1_0, p=1) + torch.norm(curr_X2 - X2_0, p=1)

            loss = clust_loss * lamda + norm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter+1) % 50 == 0:
                print('Iteration {}, Total loss:{:.8f}, clust loss:{:.8f}, L1 penalty:{:.8f}'.format(iter, loss.item(), clust_loss.item(), norm_loss.item()))

        if use_abs:
            rel_score1 = torch.mean(torch.abs(curr_X1 - X1_0), dim=0)
            rel_score2 = torch.mean(torch.abs(curr_X2 - X2_0), dim=0)
        else:
            rel_score1 = torch.mean(curr_X1 - X1_0, dim=0)
            rel_score2 = torch.mean(curr_X2 - X2_0, dim=0)
        return rel_score1.data.cpu().numpy(), rel_score2.data.cpu().numpy()
