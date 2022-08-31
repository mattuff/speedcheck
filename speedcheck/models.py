
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()

    def forward(self, x_q, x_k, x_v):
        b, c, t, h, w = x_q.size()

        r_q = x_q.permute((0, 2, 1, 3, 4)).reshape(b * t, c, h * w)
        r_k = x_k.permute((0, 2, 1, 3, 4)).reshape(b * t, c, h * w)
        r_v = x_v.permute((0, 2, 1, 3, 4)).reshape(b * t, c, h * w)

        m_c = T.matmul(r_q, r_k.transpose(1, 2))
        m_c = F.softmax(m_c, dim=0)
        a_c = T.matmul(m_c, r_v)

        m_s = T.matmul(r_q.transpose(1, 2), r_k)
        m_s = F.softmax(m_s, dim=0)
        a_s = T.matmul(m_s, r_v.transpose(1, 2)).transpose(1, 2)

        a = a_c + a_s
        a = a.reshape(b, t, c, h, w).permute((0, 2, 1, 3, 4))
        return a


class TemporalAttention(nn.Module):

    def __init__(self):
        super(TemporalAttention, self).__init__()

    def forward(self, x, x_v):
        b, c, t, h, w = x.size()

        r_q = x.permute((0, 2, 1, 3, 4)).reshape(b, t, c * h * w)
        r_k = r_q.transpose(1, 2)
        r_v = x_v.permute((0, 2, 1, 3, 4)).reshape(b, t, c * h * w)

        m_t = T.matmul(r_q, r_k)
        m_t = F.softmax(m_t, dim=0)
        a_t = T.matmul(m_t, r_v)

        a = a_t.reshape(b, t, c, h, w).permute((0, 2, 1, 3, 4))
        return a


class SSA(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(SSA, self).__init__()

        self.x_q_embed = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, 1))
        self.x_k_embed = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, 1))
        self.x_v_embed = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, 1))
        self.spatial = SpatialAttention()

        self.X_embed = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.temporal = TemporalAttention()

    def forward(self, x):
        '''
          x = T.tensor of shape (B, C, T, H, W)
        '''

        x_q = F.relu(self.x_q_embed(x))
        x_k = F.relu(self.x_k_embed(x))
        x_v = F.relu(self.x_v_embed(x))
        s = self.spatial(x_q, x_k, x_v)

        X = F.relu(self.X_embed(s))
        t = self.temporal(X, x_v)

        return t


class IdBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, kernel_size):
        super(IdBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_dim, hidden_dim, 1)
        self.norm1 = nn.InstanceNorm3d(hidden_dim)

        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.InstanceNorm3d(hidden_dim)

        self.conv3 = nn.Conv3d(hidden_dim, in_dim, 1)
        self.norm3 = nn.InstanceNorm3d(in_dim)

    def forward(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = x + x_shortcut
        x = F.relu(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, kernel_size, stride=1):
        super(ConvBlock, self).__init__()

        self.conv0 = nn.Conv3d(in_dim, out_dim, 1, stride=stride)
        self.norm0 = nn.InstanceNorm3d(out_dim)

        self.conv1 = nn.Conv3d(in_dim, hidden_dim, 1, stride=stride)
        self.norm1 = nn.InstanceNorm3d(hidden_dim)

        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.InstanceNorm3d(hidden_dim)

        self.conv3 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.norm3 = nn.InstanceNorm3d(out_dim)

    def forward(self, x):
        x_shortcut = self.conv0(x)
        x_shortcut = self.norm0(x_shortcut)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = x + x_shortcut
        x = F.relu(x)

        return x


class SSABlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, kernel_size):
        super(SSABlock, self).__init__()

        self.conv1 = nn.Conv3d(in_dim, hidden_dim, 1)
        self.norm1 = nn.InstanceNorm3d(hidden_dim)

        self.ssa2 = SSA(hidden_dim, hidden_dim)

        self.conv3 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.norm3 = nn.InstanceNorm3d(hidden_dim)

        self.conv4 = nn.Conv3d(hidden_dim, in_dim, 1)
        self.norm4 = nn.InstanceNorm3d(in_dim)

    def forward(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.ssa2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)

        x = x + x_shortcut
        x = F.relu(x)

        return x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1a', nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))),
            ('conv1b', nn.InstanceNorm3d(64)),
            ('conv1c', nn.ReLU()),
            ('conv1d', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
        ]))

        self.res2 = nn.Sequential(OrderedDict([
            ('res2a', ConvBlock(64, 64, 256, kernel_size=(1, 3, 3))),
            ('res2b', SSABlock(256, 64, kernel_size=(1, 3, 3))),
            ('res2c', SSABlock(256, 64, kernel_size=(1, 3, 3))),
        ]))

        self.res3 = nn.Sequential(OrderedDict([
            ('res3a', ConvBlock(256, 128, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2))),
            ('res3b', SSABlock(512, 128, kernel_size=(1, 3, 3))),
            ('res3c', SSABlock(512, 128, kernel_size=(1, 3, 3))),
            ('res3d', SSABlock(512, 128, kernel_size=(1, 3, 3))),
        ]))

        self.res4 = nn.Sequential(OrderedDict([
            ('res4a', ConvBlock(512, 256, 1024, kernel_size=(3, 3, 3), stride=(1, 2, 2))),
            ('res4b', IdBlock(1024, 256, kernel_size=(3, 3, 3))),
            ('res4c', IdBlock(1024, 256, kernel_size=(3, 3, 3))),
            ('res4d', IdBlock(1024, 256, kernel_size=(3, 3, 3))),
            ('res4e', IdBlock(1024, 256, kernel_size=(3, 3, 3))),
            ('res4f', IdBlock(1024, 256, kernel_size=(3, 3, 3))),
        ]))

        self.res5 = nn.Sequential(OrderedDict([
            ('res5a', ConvBlock(1024, 512, 2048, kernel_size=(3, 3, 3), stride=(2, 2, 2))),
            ('res5b', IdBlock(2048, 512, kernel_size=(3, 3, 3))),
            ('res5c', IdBlock(2048, 512, kernel_size=(3, 3, 3))),
        ]))

        self.fc6 = nn.Sequential(OrderedDict([
            ('fc6a', nn.AvgPool3d(kernel_size=(1, 7, 7))),
            ('fc6b', nn.Flatten(start_dim=1)),
            ('fc6c', nn.Dropout(p=0.3)),
            ('fc6d', nn.Linear(8192, 1024)),
            ('fc6e', nn.ReLU()),
            ('fc6f', nn.Dropout(p=0.3)),
            ('fc6g', nn.Linear(1024, 1)),
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.fc6(x)

        return x

