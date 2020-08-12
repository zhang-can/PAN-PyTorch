# Code for paper:
# [Title]  - "PAN: Towards Fast Action Recognition via Learning Persistence of Appearance"
# [Author] - Can Zhang, Yuexian Zou, Guang Chen, Lei Gan
# [Github] - https://github.com/zhang-can/PAN-PyTorch

import torch
from torch import nn
import math

class PA(nn.Module):
    def __init__(self, n_length):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,8,7,1,3)
        self.n_length = n_length
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1))
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1)
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        PA = d.view(-1, 1*(self.n_length-1), h, w)
        return PA

class VAP(nn.Module):
    def __init__(self, n_segment, feature_dim, num_class, dropout_ratio):
        super(VAP, self).__init__()
        VAP_level = int(math.log(n_segment, 2))
        print("=> Using {}-level VAP".format(VAP_level))
        self.n_segment = n_segment
        self.VAP_level = VAP_level
        total_timescale = 0
        for i in range(VAP_level):
           timescale = 2**i
           total_timescale += timescale
           setattr(self, "VAP_{}".format(timescale), nn.MaxPool3d((n_segment//timescale,1,1),1,0,(timescale,1,1)))
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.TES = nn.Sequential(
            nn.Linear(total_timescale, total_timescale*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_timescale*4, total_timescale, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.pred = nn.Linear(feature_dim, num_class)
        
        # fc init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        _, d = x.size()
        x = x.view(-1, self.n_segment, d, 1, 1).permute(0,2,1,3,4)
        x = torch.cat(tuple([getattr(self, "VAP_{}".format(2**i))(x) for i in range(self.VAP_level)]), 2).squeeze(3).squeeze(3).permute(0,2,1)
        w = self.GAP(x).squeeze(2)
        w = self.softmax(self.TES(w))
        x = x * w.unsqueeze(2)
        x = x.sum(dim=1)
        x = self.dropout(x)
        x = self.pred(x.view(-1,d))
        return x

