from redunet import *
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt

from redunet import *
import evaluate
import functional as F 
import utils
import plot


class swizzDataSet(Dataset):

    def __init__(self,x,y):
        self.x = x
        self.y = y
        pass

    def __getitem__(self,index): return x[index],y[index]
    def __len__(self):
        return len(x)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='choice of dataset')
parser.add_argument('--arch', type=str, required=True, help='choice of architecture')
parser.add_argument('--samples', type=int, required=True, help="number of samples per update")
parser.add_argument('--tail', type=str, default='', help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/', help='base directory for saving.')
parser.add_argument('--data_dir', type=str, default='./data/', help='base directory for saving.')
parser.add_argument('--batch_size', type=int, default=100, help='batch size for evaluation')
parser.add_argument('--loss', default=False, action='store_true', help='set to True if plot loss')
parser.add_argument('--ns', type=float, default=1, help='ns')
parser.add_argument('--eta', type=float, default=0.5, help='eta')
parser.add_argument('--lamda', type=float, default=500, help='lambda')
parser.add_argument('--eps', type=float, default=500, help='epsilon')
args = parser.parse_args()
    
def flatten(layers, num_classes):
    net = ReduNet(
        *[Vector(eta=args.eta, 
                    eps=args.eps, 
                    lmbda=args.lamda, 
                    num_classes=num_classes, 
                    dimensions=2
                    ) for _ in range(layers)],
    )
    return net


## CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## Model Directory
model_dir = os.path.join(args.save_dir, 
                         'forward',
                         f'{args.data}+{args.arch}',
                         f'samples{args.samples}'
                         f'{args.tail}')
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

## Data
n = 70000
ns = args.ns
r = np.random.uniform(0,ns,n)
l = np.random.uniform(0,ns,n)
t = (3 * np.pi) / 2 * (1 + 2 * r)
x = t * np.cos(t)
y = 2 * l
z = t * np.sin(t) 
u1 = np.vstack((x,z)).T
x1 = u1
y1 = torch.zeros(u1.shape[0])
theta = 30
theta = theta / (2 * np.pi)
M = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
u2 = u1 @ M 
x2 = u2
y2 = torch.ones(u2.shape[0])
x_all = torch.cat((torch.tensor(x1),torch.tensor(x2)),dim=0)
y_all = torch.cat((y1,y2),dim=0)
indexes = torch.randperm(x_all.shape[0])
x_all = x_all[indexes]
y_all = y_all[indexes]
x_train = x_all[:n//7*6]
y_train = y_all[:n//7*6]
x_test = x_all[n//7*6:]
y_test = y_all[n//7*6:]

#trainset = swizzDataSet(x_train,y_train)
#test = swizzDataSet(x_test,y_test)
num_classes = 2
#X_train, y_train = F.get_samples(trainset, args.samples)
#X_train, y_train = X_train.to(device), y_train.to(device)
X_train = x_train.float()
#print(X_train.shape)
X_test = x_test.float()
y_train = y_train.long()
y_test = y_test.long()


## Architecture
net = flatten(layers=50,num_classes=2)
#print(net)
net = net.to(device)
net = net.float()


## Training
with torch.no_grad():
    Z_train = net.init(X_train, y_train)
    losses_train = net.get_loss()
    X_train, Z_train = F.to_cpu(X_train, Z_train)
    #Z_test = net.update(X_test,y_test) 
    Z_test = net.batch_forward(X_test, batch_size=args.batch_size, loss=args.loss, device=device)

lenx = x_train.shape[0]
u1 = []
u2 = []
ux1 = []
ux2 = []
for i in range(0,lenx):
    if y_test[i] < 1e-5:
        u1.append(Z_test[i].detach().numpy())
        ux1.append(X_test[i].detach().numpy())
    else:
        u2.append(Z_test[i].detach().numpy())
        ux2.append(X_test[i].detach().numpy())

u1 = np.array(u1)
u2 = np.array(u2)
ux1 = np.array(ux1)
ux2 = np.array(ux2)
plt.subplot(1,2,1)
plt.scatter(u1[:,0],u1[:,1])
plt.scatter(u2[:,0],u2[:,1])
plt.subplot(1,2,2)
plt.scatter(ux1[:,0],ux1[:,1])
plt.scatter(ux2[:,0],ux2[:,1])
plt.show()




## Saving
utils.save_loss(model_dir, 'train', losses_train)
utils.save_ckpt(model_dir, 'model', net)

## Plotting
plot.plot_loss_mcr(model_dir, 'train')

print(model_dir)



