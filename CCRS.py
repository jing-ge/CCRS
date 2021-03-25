import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.utils import k_hop_subgraph,subgraph,contains_isolated_nodes
import networkx as nx
from torch_geometric.data import DataLoader,Batch
import warnings
 
warnings.filterwarnings('ignore')

class alexnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CRCNN(nn.Module):
    def __init__(self,n_in,n_out,n_img):
        super().__init__() 
        self.n_img = n_img
        self.lin1 = nn.Linear(n_in, n_img*3)
        self.lin2 = nn.Linear(n_img*3,n_img*n_img*3)
        self.cnn = alexnet()
        self.linout = nn.Linear(1000,n_out)
    def forward(self, x,epoch=0):
        batch = x.shape[0]
        x = self.lin1(x)
        x = F.dropout(x)
        x = self.lin2(x)
        x = F.dropout(x)
        x = x.reshape(batch,3,self.n_img,self.n_img)
        if epoch==500:
            np.save("image.npy",x.cpu().detach().numpy())
        x = self.cnn(x)
        x = self.linout(x)
        return x

def getncom(G):
        c = 0
        for i in nx.connected_components(G):
            c += 1
        return c
def product2edgeindex(product):
    value = 0.9
    while True:
        idx = np.where(product>value)
        # index =
        res = contains_isolated_nodes(torch.Tensor(idx).long(), num_nodes=39)
        G=nx.Graph()
        G.add_edges_from(np.array(idx).T.tolist())
        x = getncom(G)
        if x==1:
            return  np.unique(np.array([np.hstack([idx[0],idx[1]]),np.hstack([idx[1],idx[0]])],dtype=np.int),axis=1).astype(np.int)
        value -= 0.1

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

def bCCRGNN(tensorinput):
    model = torch.load("/home/fengbojing/fengbojing/CCRS/CCRGNN.pt").cuda()
    model.eval()
    tensorinput.resize([39,1])
    product = np.exp(tensorinput.dot(tensorinput.T))
    product = noramlization(product)
    idx = product2edgeindex(product)
    graph = Data(edge_index = torch.Tensor(idx).long(), x = torch.Tensor(tensorinput))
    loader = DataLoader([graph], batch_size=1)
    for batch in loader:
        out = model(batch)
    out = out.cpu()
    out = int(torch.argmax(F.softmax(out,1), 1).detach().numpy())
    m = ["AAA","AA", "A","BBB","BB","B","CCC","CC","C"]
    level = m[out]
    return level
def bCCRCNN(x):
    model = torch.load("/home/fengbojing/fengbojing/CCRS/model_CCRCNN.pkl")
    model.eval()
    tensorinput = torch.Tensor(x).unsqueeze(0)
    out = model(tensorinput)
    out = int(torch.argmax(F.softmax(out,1), 1).detach().numpy())
    m = ["AAA","AA", "A","BBB","BB","B","CCC","CC","C"]
    level = m[out]
    return level

def bASSL4CCR(x):
    model = torch.load("/home/fengbojing/fengbojing/CCRS/ASSL4CCR.pt")
    model.eval()
    tensorinput = torch.Tensor(x).unsqueeze(0)
    out = int(model("predict",x = tensorinput))
    # out = int(torch.argmax(F.softmax(out,1), 1).detach().numpy())
    m = ["AAA","AA", "A","BBB","BB","B","CCC","CC","C"]
    level = m[out]
    return level
x = np.random.rand(39)
std = np.array([5.27138270e-01, 8.25752230e+00, 9.95533666e-01, 2.26933173e+09,
    1.01670778e+00, 7.87983283e+09, 6.64906142e+09, 1.30192999e+02,
    1.38856619e+03, 1.78234935e+03, 5.24114252e+03, 5.14663385e-01,
    2.39990120e-01, 1.73779538e+00, 2.79022304e-01, 6.83370576e+01,
    6.80914889e+00, 7.50007005e+00, 1.54219387e+00, 1.35281783e+00,
    6.63452952e-01, 5.58468562e-01, 2.47359585e+00, 3.68025702e+00,
    2.74342540e+00, 3.71265192e-01, 6.53042120e-01, 2.58621214e+02,
    5.98361612e+00, 7.29341857e+01, 3.72839459e+00, 5.27138270e-01,
    1.50166855e+01, 2.39990120e-01, 4.33075673e+00, 8.60610288e+00,
    1.74420606e+01, 1.48476654e+01, 1.59381980e+00])
mean = np.array([ 4.49396750e-04, -2.22269479e-01,  2.72138836e-01,  5.33585321e+08,
    3.28045229e-01,  2.80571117e+09,  2.35327421e+09,  1.86014395e+01,
    1.70753112e+02,  7.17274753e+01,  7.16284455e+02,  5.83205251e-01,
    2.64915407e-01, -6.14608971e-03,  1.05549025e-01, -3.06572374e+00,
    -4.43146209e-01, -4.33107792e-01,  1.63866649e+00,  1.24622301e+00,
    4.91154084e-01,  1.71464830e-01,  6.29593370e-01,  2.83910118e+00,
    6.00980812e-01,  4.92503620e-01,  3.34616178e-01,  2.53245673e+01,
    2.26663399e-02,  1.44388696e+00,  1.17646309e-01,  4.49396750e-04,
    3.29824905e+00,  2.64915407e-01,  1.21270507e+00, -3.27024112e-01,
    3.91242440e-01,  1.00338497e+00,  3.64050784e-02])
tensorinput = (x-mean)/(std)

# asslout = bASSL4CCR(tensorinput)
# print(asslout)
# cnnout = bCCRCNN(tensorinput)
# print(cnnout)
# gnnout = bCCRGNN(tensorinput)
# print(gnnout)

from flask import Flask
from flask import request

import json

app = Flask(__name__)

@app.route('/CCRS',methods=['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        que = json.loads(request.get_data())
        print(que)
        model = que['model']
        tensorinput = (np.array(que['x'])-mean)/(std)
        asslout = bASSL4CCR(tensorinput)
        print(asslout)
        cnnout = bCCRCNN(tensorinput)
        print(cnnout)
        gnnout = bCCRGNN(tensorinput)
        print(gnnout)
        if model == "ASSL4CCR":
            return asslout
        elif model == "CCRGNN":
            return gnnout
        elif model=="CCRCNN":
            return cnnout
        else:
            return 'model error!!! select from CCRCNN,CCRGNN,ASSL4CCR'
    else:
        return "error"

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)