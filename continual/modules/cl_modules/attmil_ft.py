import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


from modules.mil_modules.attmil import initialize_weights


class DAttention(nn.Module):
    def __init__(self,input_dim,n_classes,dropout,act,rrt=None):
        super(DAttention, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(input_dim, 512)]
        
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]
        if rrt is not None:
            self.feature += [rrt] 
        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.ModuleList([
            nn.Linear(self.L*self.K, n_classes),
        ])
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_attn=False,no_norm=False):

        feature = self.feature(x)

        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL

        Y_prob = []
        for linear_m in self.classifier:
            logits = linear_m(M)
            Y_prob.append(logits)

        Y_prob = torch.cat(Y_prob, dim=-1)

        # print(Y_prob.shape)


        if return_attn:
            if no_norm:
                return Y_prob,A_ori
            else:
                return Y_prob,A
        else:
            return Y_prob
