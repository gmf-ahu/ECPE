import math
import os
from pickle import TRUE
from re import T
import time
from xml.sax.handler import property_lexical_handler
import numpy as np
from sympy import Id, div, false, true
import torch
from torch import device, nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from utils import warp_tqdm
from matplotlib.ticker import LogLocator


def groups(xs, xq, ys, yq, runs, ways, shot, queries, device, config, terminal=True,entro=True,pro=True,dataset=None):
	xs, xq, ys, yq = xs.to(device), xq.to(device), ys.to(device), yq.to(device)
	balance = False if config.dirichlet !=0 else True
	model = Groups(runs, ways, shot, queries, ys,config.lam,config.alpha,config.link_rate,
						  config.update_rate,config.k_hop,config.signal, config.epochs,config.omega,terminal,balance,config.entro,config.pro,config.dataset).to(device)
	result = model(xs, xq, ys, yq)

	return result


class Groups(nn.Module):
    def __init__(self,runs, ways, shot, queries, ys, lam=10,
                 alpha=0.7, link_rate=0.4, update_rate=0.6,
                 k_hop=4,signal=0.5, epochs=10,omega=0.0008, terminal=True,balance=True,entro=True,pro=True,dataset=None):
        super().__init__()
        self.runs = runs
        self.ways = ways
        self.queries = queries
        self.shot = shot
        self.proto = None
        self.lam = lam
        self.ns = ways * shot
        self.qs = ways * queries
        self.ys = ys
        self.eps = 1e-8
        self.alpha = alpha
        self.link_rate = link_rate
        self.update_rate = update_rate
        self.epochs = epochs
        self.terminal = terminal
        self.balance = balance
        self.dataset = dataset
    
    def forward(self, xs, xq, ys, yq):
        features = torch.cat((xs, xq), dim=1)
        labels = torch.cat((ys, yq), dim=1)
        prob_query = self.predict(features,labels)
        return prob_query
    
    
    def predict(self, features, labels):
        device = features.device
        samples = features.shape[1]

        self.proto = features[:, :self.ns].reshape(self.runs, self.ways, self.shot, -1).mean(dim=2) #更新原型
        

        Iden = torch.eye(samples, device=device).unsqueeze(0)
        dist = self.get_distance(features,features)
        W = self.build_graph(dist,self.link_rate)
        Inverse_W = torch.inverse(Iden - self.alpha * W)

        Z = self.get_prob(features, self.proto, True)

        for i in warp_tqdm(range(int(self.epochs)), not self.terminal):
            # Build Graph
            # Get Y in the Algorithm
            features = features[:,:self.ns+ self.qs]
            Z = self.get_prob(features, self.proto, True)
            


            entropy = compute_entropy(Z)
            Z = Z * (1-entropy.unsqueeze(-1))
            # Soft Label Propagation

            self.update_prototype(Z, features, self.update_rate)


            # features = features[:,:self.ns+ self.qs]
            dist_proto = self.get_distance(self.proto,self.proto) # [10000,5,5]
            score = compute_entropy(dist_proto) # [10000,5]
            mask = torch.zeros_like(score)# [10000,5]
            omega = entropy.mean(1,keepdim=True)
            mask[score<omega] = 1# [10000,5]
            add_proto = self.proto * mask.unsqueeze(-1) # [10000,5,40]
            dist_proto = dist_proto * mask.unsqueeze(-1)
            features = torch.cat((features,add_proto),dim=1)# [10000,85,40]
            Z = torch.cat((Z,dist_proto),dim=1)
            
            # #**********
            Iden = torch.eye(features.shape[1], device=device).unsqueeze(0)
            dist = self.get_distance(features,features)
            W = self.build_graph(dist,self.link_rate)
            Inverse_W = torch.inverse(Iden - self.alpha * W)
            #******


            
            Z = Inverse_W  @  Z
            # Normalize(if imbalance, we only use Z=Z/Z.sum(-1))

            Z = compute_optimal_transport(Z, self.ns, self.ys, 1e-3, self.balance)


        # get final accuracy and return it
        features = features[:,:self.ns+ self.qs]
        op_xj = Z[:,:self.ns+self.qs]

        # op_xj = self.get_prob(features, self.proto)
        
        olabels = op_xj.argmax(dim=2)
        matches = labels.eq(olabels).float()

        acc_test = matches[:, self.ns:].mean(1)

        torch.save(acc_test,'mani.pth')
        high = torch.argsort(acc_test,descending=True)
        return acc_test
    
    

    def get_prob(self, features, proto, iter=False):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = torch.cdist(features, proto).pow(2)

        P = torch.zeros_like(dist)
        Pq = dist[:, self.ns:]
        Pq = torch.exp(- self.lam * Pq)
        Pq = compute_optimal_transport(Pq, 0, self.ys, 1e-3,self.balance)
        P[:, self.ns:] = Pq
        P[:, :self.ns].scatter_(2, self.ys.unsqueeze(2), 1)
        return P


    def get_distance(self, features1,features2):
        # get pairwise distance of samples
        dist = torch.cdist(features1, features2).pow(2)
        dist = torch.exp(- dist * self.lam)
        return dist



    def build_graph(self,W, link_rate=0.4):
        samples = W.shape[1]
        Iden = torch.eye(samples, device=W.device).unsqueeze(0)
        # Set 0 to Diagnose.
        W = W * (1 - Iden)

        # Sparsify the adjacency matrix.
        # W = keep_top_k_row(W, int(link_rate))
        # Symmetrically normalize
        D = W.sum(-1).pow(-0.5)
        W = D.unsqueeze(-2) * W * D.unsqueeze(-1)
        # W = (W+W.transpose(-2,-1))/2

        return W

    def update_prototype(self, transport, features, alpha):
        new_proto = transport.permute(0, 2, 1).matmul(features).div(transport.sum(dim=1).unsqueeze(2))
        self.proto = (1-alpha) * self.proto + alpha * new_proto

        
        
def compute_entropy(prob_matrix, normalize=True):

    epsilon = 1e-12
    prob_matrix = prob_matrix + epsilon  
    prob_matrix = prob_matrix / prob_matrix.sum(dim=2, keepdim=True)  

    entropy = -torch.sum(prob_matrix * torch.log(prob_matrix), dim=2)


    if normalize:
        max_entropy = torch.log(torch.tensor(prob_matrix.size(2), dtype=prob_matrix.dtype))  # log(C)
        entropy = entropy / max_entropy

    return entropy


def compute_optimal_transport(M, n_lsamples, labels, epsilon=1e-6,class_balance=True):
    # r : [runs, total samples], c : [runs, ways]
    # n samples, m ways
    n_runs, n, ways = M.shape
    r = torch.ones(n_runs, n, device=M.device)
    c = torch.ones(n_runs, ways, device=M.device) * n // ways
    u = torch.zeros(n_runs, n, device=r.device)
    P = M
    maxiters = 1000
    iters = 1

    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    while torch.max(torch.abs(u - P.sum(2))) > epsilon:
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        if class_balance:
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
        P[:, :n_lsamples].fill_(0)
        P[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
        if iters == maxiters:
            break
        iters = iters + 1
    return P




