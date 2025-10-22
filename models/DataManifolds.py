import math
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from utils import warp_tqdm


# 这是我的方法，用到的更多一些

def ManiFolds(xs, xq, ys, yq, runs, ways, shot, queries, device, config, terminal=True):
	xs, xq, ys, yq = xs.to(device), xq.to(device), ys.to(device), yq.to(device)
	balance = False if config.dirichlet !=0 else True
	# 比如这里用到了alpha，link，update，kop
	# 都是通过前面传过来的
	model = DataManifolds(runs, ways, shot, queries, ys,config.lam,config.alpha,config.link_rate,
						  config.update_rate,config.k_hop,config.signal, config.epochs,terminal,balance).to(device)
	result = model(xs, xq, ys, yq)

	return result


class DataManifolds(nn.Module):
	def __init__(self, runs, ways, shot, queries, ys, lam=10,
				 alpha=0.7, link_rate=0.4, update_rate=0.6,
				 k_hop=4,signal=0.5, epochs=10, terminal=True,balance=True):
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
	def forward(self, xs, xq, ys, yq):
		features = torch.cat((xs, xq), dim=1)
		labels = torch.cat((ys, yq), dim=1)

		prob_query = self.preidct(features, labels)
		return prob_query

	def preidct(self, features, labels):
		device = features.device
		samples = features.shape[1]
		Iden = torch.eye(samples, device=device).unsqueeze(0)

		# tSNE(features[0],labels[0],5)

		# tSNE(features[13], labels[0], 5,f'{self.ways}-Way {self.shot}-Shot w/o JMP')

		dist = self.get_distance(features,features)
		features = self.graph_convolution(features,dist,self.k_hop)
		# print(features.shape)
		self.proto = features[:, :self.ns].reshape(self.runs, self.ways, self.shot, -1).mean(dim=2)
		# print(f'Prototype Shape:\t{self.proto.shape}')

		# tSNE(features[13],labels[0], 5,f'{self.ways}-Way {self.shot}-Shot w/ JMP')

		dist = self.get_distance(features,features)
		W = self.build_graph(dist,self.link_rate)
		Inverse_W = torch.inverse(Iden - self.alpha * W)

		for i in warp_tqdm(range(int(self.epochs)), not self.terminal):
			# Build Graph
			# Get Y in the Algorithm
			Z = self.get_prob(features, self.proto,True)

			# # Vanilla Lable Propagation
			# Y = torch.zeros_like(Z)
			# Y[:, :self.ns].scatter_(2, self.ys.unsqueeze(2), 1)
			# Z = Inverse_W @ Y

			# Soft Lable Propagation
			Z = Inverse_W  @ Z

			Z= torch.clamp(Z,0)

			# Normalize(if imbalance, we only use Z=Z/Z.sum(-1)）
			Z = compute_optimal_transport(Z, self.ns, self.ys, 1e-3, self.balance)
			# update Prototype
			self.update_prototype(Z, features, self.update_rate)

		# vis_idx = 7787 #imbalance idx
		# vis_idx = 6234
		# tSNE(features[vis_idx], labels[0], 5, self.proto[vis_idx], self.shot)

		# get final accuracy and return it
		op_xj = self.get_prob(features, self.proto)
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
		return P

	def graph_convolution(self,X, A, k=1.0):
		samples = X.shape[1]
		Iden = torch.eye(samples, device=X.device).unsqueeze(0)
		D = A.sum(-1).pow(-0.5)
		W = D.unsqueeze(1) * A * D.unsqueeze(-1)
		L = Iden - W
		G = (Iden - 0.5 * L).pow(k)

		X = G @ X
		return X

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
		W = keep_top_k_row(W, int(link_rate))
		# Symmetrically normalize
		D = W.sum(-1).pow(-0.5)

		W = D.unsqueeze(-2) * W * D.unsqueeze(-1)

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


def conf_matrix(predictions,labels):
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
	plt.savefig(f'visual/tsne_{shot}.pdf')
	# plt.axis('off')
	plt.show()
