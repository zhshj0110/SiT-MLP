import pdb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def linear_init(l):
    if l.weight is not None:
        nn.init.trunc_normal_(l.weight, std=.02)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)





class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)



    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class  unit_gmlp(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, residual=True, heads=4, ):
        super(unit_gmlp, self).__init__()
        self.heads = heads
        # channel projection
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels*2, kernel_size=1)
        self.activation = nn.GELU()
        # aggreation
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.spatial_weight = nn.Parameter(torch.zeros(heads, num_point,num_point), requires_grad=True)
        self.shared_topology = nn.Parameter(torch.stack([torch.eye(num_point)]*heads), requires_grad=True)
        # update
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(self.bn(x)))
        z1, z2 = x.chunk(2, dim=1)  # (B, C, T, V)
        z1 = rearrange(self.bn1(z1), 'b (h d) t v -> b h d t v', h=self.heads)
        z2 = rearrange(self.bn2(z2), 'b (h d) t v -> b h d t v', h=self.heads)
        y1 = torch.einsum('huv,bhdtv->bhdtu', self.shared_topology, z1)  # topology connection
        attn = torch.einsum('huv,bhdtv->bhdtu', self.spatial_weight, z2)  # topology node attention
        y2 = z1 * attn
        y = y1 + y2
        y = rearrange(y, 'b h d t v -> b (h d) t v')
        y = self.conv2(y)
        y += self.down(residual)
        y = self.relu(y)
        return y

class MLP_TCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_joint, stride=1, residual=True, kernel_size=5, dilations=[1,2]):
        super(MLP_TCN_unit, self).__init__()

        self.gmlp = unit_gmlp(in_channels, in_channels, num_joint)
        self.tcn = MultiScale_TemporalConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
    
    
    def forward(self, x):
        y = self.relu(self.tcn(self.gmlp(x)) + self.residual(x))

        return y

class Model(nn.Module):
    def __init__(self, 
                 num_class=60, 
                 num_point=25,
                 num_person=2,
                 in_channels=3,
                 graph=None,
                 k=0,
                 num_head=4
                 ):
        super(Model, self).__init__()

        base_channels = 64
        self.num_class = num_class
        self.num_point = num_point
        self.A_vector = self.get_A(graph, k)
        self.to_joint_embedding = nn.Linear(in_channels, base_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, base_channels))
        self.data_bn = nn.BatchNorm1d(num_person * base_channels * num_point)

        self.l1 = MLP_TCN_unit(base_channels, base_channels, num_point)
        # self.l2 = MLP_TCN_unit(base_channels, base_channels, num_point)
        # self.l3 = MLP_TCN_unit(base_channels, base_channels, num_point)
        self.l4 = MLP_TCN_unit(base_channels, base_channels*2, num_point, stride=2)
        self.l5 = MLP_TCN_unit(base_channels*2, base_channels*2, num_point)
        # self.l6 = MLP_TCN_unit(base_channels*2, base_channels*2, num_point)
        self.l7 = MLP_TCN_unit(base_channels*2, base_channels*4, num_point, stride=2)
        # self.l8 = MLP_TCN_unit(base_channels*4, base_channels*4, num_point)
        self.l9 = MLP_TCN_unit(base_channels*4, base_channels*4, num_point)
        self.fc = nn.Linear(base_channels*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
    

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(self.num_point)
        if k == 0:
            return torch.tensor(I, dtype=torch.float32)
        return  torch.tensor(I - np.linalg.matrix_power(A_outward, k), dtype=torch.float32)
    
    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = self.to_joint_embedding(x)
        x += self.pos_embedding
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        x = self.l7(x)
        # x = self.l8(x)
        x = self.l9(x)
        C_new = x.size(1)
        x = x.reshape(N, M, C_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)