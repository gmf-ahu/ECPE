from sympy import primorial
import torch
from torch import nn
batch, node, dim = 1, 5, 10
n_cls = 5
l1=2
a = torch.rand(batch,node,dim)
print(f'Feature {a.shape}')

class ConvNet(nn.Module):
	def __init__(self, dim):
		super().__init__()
		hidden_dim = dim//2
		self.conv = nn.Linear(dim,hidden_dim)
		self.act = nn.ReLU()
		self.norm = nn.LayerNorm(hidden_dim)
		self.conv2 = nn.Linear(hidden_dim,dim)
		self.norm2 = nn.LayerNorm(dim)
		self.head = nn.Linear(dim,n_cls)
		self.softmax = nn.Softmax(dim=-1)
	
	def forward(self, x):
		# x/res: [b, n, dim]
		res = x
		# x: [b, n, hidden_dim]
		x = self.conv(x)
		x = self.act(x)
		x = self.norm(x)
		# x: [b, n, dim]
		x = self.conv2(x)
		x = self.act(x)
		x = self.norm2(x) + res
		# x: [b, n, n_cls]
		x = self.head(x)
		x = self.softmax(x)
		return x
		pass


def train(x):
	loss = nn.CrossEntropyLoss()
	# 
	net = ConvNet(dim)
	print(net)
	# 现有的 Label，需要用scatter变换
	label = torch.randint(0,5,(batch,node))
	ys = label[:,:l1]
	print(f"ys: {ys} \n label: {label}")
	Ys = torch.zeros(batch,l1,n_cls)
	print("Ys: ",Ys)
	Ys.scatter_(-1,ys.unsqueeze(-1),1)
	print(Ys)
	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	Y = torch.zeros(batch,node,n_cls)
	Y.scatter_(-1,label.unsqueeze(-1),1)
	print(Y)



	num_epochs = 1000
	for epoch in range(num_epochs):
		optimizer.zero_grad()
		# pred: [b, n , nls],包含了映射以及标签传播完整过程
		pred = net(x)
		pred = pred[:,:4]
		print(pred.shape)
		print(Ys.shape)
		l = loss(pred,Ys)
		l.backward()
		optimizer.step()
	# 
		# print(f'epoch {epoch + 1}, loss {l:f}')
	pred = net(x)
	l = loss(pred, Y)
	print(l)
	print(Y)
	print(pred)

train(a)