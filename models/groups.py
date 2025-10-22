import math
import os
from pickle import TRUE
from re import T
import time
from xml.sax.handler import property_lexical_handler
import numpy as np
from sympy import Id, div, false
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch import device, nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from utils import warp_tqdm
from matplotlib.ticker import LogLocator

# 这是我的方法，用到的更多一些

def groups(xs, xq, ys, yq, runs, ways, shot, queries, device, config, terminal=True,entro=True,pro=True,dataset=None):
	xs, xq, ys, yq = xs.to(device), xq.to(device), ys.to(device), yq.to(device)
	balance = False if config.dirichlet !=0 else True
	# 比如这里用到了alpha，link，update，kop
	# 都是通过前面传过来的
	model = Groups(runs, ways, shot, queries, ys,config.lam,config.alpha,config.link_rate,
						  config.update_rate,config.k_hop,config.signal, config.epochs,config.omega,terminal,balance,config.entro,config.pro,config.dataset).to(device)
	result = model(xs, xq, ys, yq)

	return result
# 假设 xs, xq, ys, yq, device 和 config 已经定义


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
        self.softmax = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-2)
        self.ns = ways * shot
        self.qs = ways * queries
        self.ys = ys
        self.eps = 1e-8
        self.label_graph = self.generate_diagonal_block_matrix(self.ns,shot,ways)
        ####
        self.alpha = alpha
        self.link_rate = link_rate
        self.update_rate = update_rate
        self.k_hop = k_hop
        ####
        self.signal = signal
        self.epochs = epochs
        self.terminal = terminal
        self.tau = 0.0001
        self.balance = balance
        self.entro = entro
        self.pro = pro
        # self.zeta = zeta
        self.omega = omega
        self.dataset = dataset
    
    def forward(self, xs, xq, ys, yq):
        features = torch.cat((xs, xq), dim=1)
        labels = torch.cat((ys, yq), dim=1)
        prob_query = self.predict(features,labels)
        # print(prob_query)
        return prob_query
    
    
    def predict(self, features, labels):
        device = features.device
        # pro = self.pro
        samples = features.shape[1]
        omega = self.omega
        # tSNE(features[0],labels[0],5)
        # tSNE(features[13], labels[0], 5,f'{self.ways}-Way {self.shot}-Shot w/o JMP')
        # print(features)
        # dist = self.get_distance(features,features)
        # features = self.graph_convolution(features,dist,self.k_hop)  #更新特征矩阵
        # print(features)
        self.proto = features[:, :self.ns].reshape(self.runs, self.ways, self.shot, -1).mean(dim=2) #更新原型
        # tSNE(features[13],labels[0], 5,f'{self.ways}-Way {self.shot}-Shot w/ JMP')
        # temp_dist = self.get_distance(features,self.proto).mean()
        # if (temp_dist) > omega:
        #     pro = False
        # else:
        #     pro = True
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
            

            # # Vanilla Label Propagation
            # Y = torch.zeros_like(Z)
            # Y[:, :self.ns].scatter_(2, self.ys.unsqueeze(2), 1)
            # Z = Inverse_W @ Y
            entropy = compute_entropy(Z)
            #entro=0
            Z = Z * (1-entropy.unsqueeze(-1))
            # Soft Label Propagation
            # ***proto = 0
            self.update_prototype(Z, features, self.update_rate)

            # #***********
            
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
            #entro=0
            Z = compute_optimal_transport(Z, self.ns, self.ys, 1e-3, self.balance)


        # get final accuracy and return it
        features = features[:,:self.ns+ self.qs]
        op_xj = Z[:,:self.ns+self.qs]

        # vis_idx = 7787 #imbalance idx
        # vis_idx = 5245
        # tSNE(features[vis_idx], labels[0], 5, self.proto[vis_idx], self.shot)
        
        
        # op_xj = self.get_prob(features, self.proto)
        ### 计算可视化图
        # figure(Z,labels,self.ys,self.dataset,self.shot)
        
        
        
        
        # conf_matrix(op_xj[vis_idx], labels[vis_idx])
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
        # if iter:
        # 	P[:, :self.ns].fill_(-self.signal)
        # else:
        # 	P[:, :self.ns].fill_(0)
        P[:, :self.ns].scatter_(2, self.ys.unsqueeze(2), 1)
        # P[:, -self.ways:] = torch.eye(self.ways).unsqueeze(0)
        return P


    def get_distance(self, features1,features2):
        # get pairwise distance of samples
        dist = torch.cdist(features1, features2).pow(2)
        dist = torch.exp(- dist * self.lam)
        return dist



    def build_graph(self,W, link_rate=0.4):
        samples = W.shape[1]
        Iden = torch.eye(samples, device=W.device).unsqueeze(0)
        # support_W = W[:,:self.ns,:self.ns]
        # mask = torch.eye(self.ns,device=W.device)[None, :, :]
        # W[:,:self.ns,:self.ns] = support_W * mask
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

    def generate_diagonal_block_matrix(self,n, k, num_blocks):
        # 创建一个全0矩阵
        matrix = torch.ones(n, n)

        # 生成对角块
        for i in range(num_blocks):
            start_idx = i * k
            end_idx = min((i + 1) * k, n)
            matrix[start_idx:end_idx, start_idx:end_idx] = 1/k
        matrix += torch.eye(n, n)*(k-1)/k

        return matrix

def figure(Z,labels,ys, dataset, shot):
    sup_len=ys.shape[1]
    # print(sup_len)
    entropy = compute_entropy(Z, False)

    entropy = entropy[100,sup_len:]

    label = labels[100][sup_len:]
    index = torch.argsort(label,descending=False)
    entropy = entropy[index]

    class_list = []
    t = 0
    for i in range(5):
        cls_ent, idx = torch.sort(entropy[t:t+15])
        class_list.append(cls_ent)
        t += 15
    entropy = torch.cat(class_list,dim=0) 

    x = np.arange(len(entropy))
    y = entropy.cpu().numpy()  # 转换为 NumPy 数组以便绘图

    # 绘制条形图
    fig, ax = plt.subplots(figsize=(10, 5))
    color_list = ['tab:blue', 'tab:green','tab:orange','tab:brown', 'tab:purple']

    t = 0
    for color in color_list:
        ax.bar(torch.arange(t+1,t+16), y[t:t+15], color=color, zorder=2, lw=0.5,width=1, edgecolor="b", alpha=0.8) #整体偏移1
        t +=15
    # 设置图表属性
    
    xticks = np.arange(0, len(x)+1, 15)  # 每隔 15 个点取一个刻度，从 0 开始
    if x[0] in xticks:  # 确保第零个刻度 (0) 被去除
        xticks = np.delete(xticks, x[0])
    if x[1] not in xticks:  # 确保第一个刻度 (1) 被包含
        xticks = np.append(xticks, x[1])
    ax.margins(x=0)
    ax.set_xticks(xticks)
    # ax.set_yscale('log')
    # ax.set_ylim(bottom=1)
    ax.set_title(f"{dataset}", fontsize=20)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Entropy", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    save_path = os.path.join("figure",f"{dataset}_{shot}shot_entropy_chart.png")
    fig.tight_layout()
    # 显示图表
    ax.figure.savefig(save_path)
        
        
def compute_entropy(prob_matrix, normalize=True):
    # 确保概率矩阵的形状为 (B, N, C) 
    # 对于每个样本，我们需要计算其熵
    # 使用epsilon以避免log(0)的情况
    epsilon = 1e-12
    prob_matrix = prob_matrix + epsilon  # 防止概率为0
    prob_matrix = prob_matrix / prob_matrix.sum(dim=2, keepdim=True)  # 确保每行和为1

    # 计算熵
    entropy = -torch.sum(prob_matrix * torch.log(prob_matrix), dim=2)


    if normalize:
    # 将熵归一化到 [0, 1]
        max_entropy = torch.log(torch.tensor(prob_matrix.size(2), dtype=prob_matrix.dtype))  # log(C)
        entropy = entropy / max_entropy

    return entropy


def softmax_tau(prob,tau,linear=True):
    if not linear:
        prob = prob/tau
        prob = F.softmax(prob,-1)
        # print(prob[0])
        return prob
    else:

        prob/= prob.sum(-1,keepdim=True)
        return prob


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


def keep_top_k_row(matrix, k):
    batch_size, num_nodes, _ = matrix.shape
    values, indices = torch.topk(matrix, k, dim=-1)
    result = torch.zeros_like(matrix)
    result.scatter_(-1,indices,values)
    return result


def conf_matrix(predictions,labels,cm):
    # 假设你有一个NxC的预测概率矩阵和N个样本的标签
    N, C = predictions.shape

    # 将PyTorch张量转换为NumPy数组
    predictions_np = predictions.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 计算混淆矩阵
    cm = confusion_matrix(labels_np, predictions_np)
    print(cm)
    # 使用Seaborn绘制混淆矩阵的热图

    fig, ax = plt.subplots()

    # 绘制热力图
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Sample count', rotation=270, labelpad=20)

    # 设置轴标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))
            # ylabel='True label',
            # xlabel='Predicted label')

    # 设置轴的刻度标签
    ax.set_xticklabels([f'{i}' for i in range(C)], rotation=45, ha='right',
                       rotation_mode='anchor')

    # 设置轴的标题
    ax.set_title('Confusion Matrix')

    # 显示图形
    plt.savefig("visual/conf_matrix.pdf")
    plt.show()

def tSNE(features, labels, num_class, proto=None, shot=None,):
    features = torch.cat((features, proto), dim=0)
    features = features.cpu()
    labels = torch.cat((labels,torch.tensor([0,1,2,3,4],device=labels.device)))
    labels = labels.cpu()

    tsne = TSNE(2)  # 降维到2维
    embedded_features = tsne.fit_transform(features)
    colors = plt.cm.rainbow(np.linspace(0, 1,num_class))
    b = torch.randperm(num_class)
    colors = colors[b]
    # marker_size = 10
    # Visualize the results
    plt.figure(figsize=(8, 7))
    for i, i_c in enumerate(range(num_class)):
        plt.scatter(embedded_features[:-5][labels[:-5] == i_c, 0],
                    embedded_features[:-5][labels[:-5] == i_c, 1],
                    color=colors[i],
                    label=str(i_c))
    for i, i_c in enumerate(range(num_class)):
        plt.scatter(embedded_features[-5:][labels[-5:] == i_c, 0],
                    embedded_features[-5:][labels[-5:] == i_c, 1],
                    color=colors[i],
                    marker= '*',
                    s = 200,
                    linewidths=0.5,
                    edgecolors='k',
                    label=str(i_c))
        # class_center = np.mean(embedded_features[labels == i], axis=0)
        # plt.scatter(class_center[0], class_center[1], color=colors[i],
        #             marker='*', s=marker_size*10, edgecolor='black', linewidth=0.1)

    plt.legend().set_visible(False)
    # plt.title(shot)
    plt.tight_layout()
    plt.savefig(f'visual/tsne_{shot}.jpg')
    # plt.axis('off')
    plt.show()
