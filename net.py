import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Encoder(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        n_hidden1 = 512
        n_hidden2 = 512
        self.model = nn.Sequential(nn.Linear(n_in,n_hidden1), nn.LayerNorm((n_hidden1,)),nn.ReLU(),nn.Dropout(0.5),
                                    nn.Linear(n_hidden1,n_hidden2), nn.LayerNorm((n_hidden2,)),nn.ReLU(),nn.Dropout(0.5),
                                    nn.Linear(n_hidden1, n_out)) 
    def forward(self,x):
        return self.model(x)
class Discriminator(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        n_hidden1 = 256
        n_hidden2 = 64
        self.model = nn.Sequential(nn.Linear(n_in,n_hidden1), nn.LayerNorm((n_hidden1,)),nn.ReLU(),nn.Dropout(0.5),
                                    nn.Linear(n_hidden1,n_hidden2), nn.LayerNorm((n_hidden2,)),nn.ReLU(),nn.Dropout(0.5),
                                    nn.Linear(n_hidden1, 2)) 
    def forward(self,x):
        return self.model(x)
class LabeledClassfier(nn.Module):
    def __init__(self,n_in,n_out):
        super().__init__()
        self.lin1 = nn.Linear(n_in, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 128)
        self.lin4 = nn.Linear(128,n_out)
    def forward(self, x):
        x = F.sigmoid(self.lin1(x))
        x = F.sigmoid(self.lin2(x))
        x = F.sigmoid(self.lin3(x))
        x = self.lin4(x)
        return x
class UnLabeledClassfier(nn.Module):
    def __init__(self, n_in,n_out):
        super().__init__()
        n_hidden1 = 1024
        n_hidden2 = 512
        n_hidden3 = 128
        self.model = nn.Sequential(nn.Linear(n_in,n_hidden1), nn.LayerNorm((n_hidden1,)),nn.ReLU(),nn.Dropout(0.5),
                                    nn.Linear(n_hidden1,n_hidden2), nn.LayerNorm((n_hidden2,)),nn.ReLU(),nn.Dropout(0.5),
                                    nn.Linear(n_hidden2,n_hidden3), nn.LayerNorm((n_hidden3,)),nn.ReLU(),nn.Dropout(0.5),
                                    nn.Linear(n_hidden3, n_out)) 
    def forward(self,x):
        return self.model(x)
class SSL4CCR(nn.Module):
    def __init__(self,n_in,n_out):
        super().__init__()
        d_encode = 256
        self.encoder = Encoder(n_in,d_encode)
        self.discriminator = Discriminator(d_encode)
        self.classfier_label = LabeledClassfier(d_encode,n_out)
        self.classfier_unlabel = LabeledClassfier(d_encode,n_out)
        self.lambda1 = 0.7
        self.lambda2 = 0.5
    def forward(self, mode,**kwargs):
        if mode== "loss":
            return self.cal_loss(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
    def cal_loss(self,**kwargs):
        #get data
        data_label = kwargs["data_label"]
        label = kwargs["label"]
        data_unlabel = kwargs["data_unlabel"]
        pseudo_lable = kwargs["pseudo_lable"]
        
        # encoding
        data_label_encode = self.encoder(data_label)
        data_unlabel_encode = self.encoder(data_unlabel)

        # supervised loss
        sup_out = self.classfier_label(data_label_encode)

        supervised_loss =  F.cross_entropy(sup_out,label)

        # unsupervised loss
        unsup_out = self.classfier_unlabel(data_unlabel_encode)
        unsupervised_loss =  F.cross_entropy(unsup_out,pseudo_lable)

        # Adversarial loss
        adversarial_loss = F.cross_entropy(torch.cat([data_label_encode,data_unlabel_encode],dim=0),
            torch.cat([torch.zeros(data_label.shape[0]),torch.ones(data_unlabel.shape[0])],dim=0).long().cuda())
        
        loss = supervised_loss + self.lambda1*unsupervised_loss + self.lambda2*adversarial_loss
        return loss
    def predict(self,x=None):
        encode = self.encoder(x)
        out = self.classfier_label(encode)
        return torch.argmax(F.softmax(out,1), 1)