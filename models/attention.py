import torch
import torch.nn as nn
import math 

class LinearPositionalEncoding(nn.Module):
    def __init__(self, width=16, height=8):
        super(LinearPositionalEncoding, self).__init__()
        pe = torch.arange(height).view(1,1,height,1).repeat(1,1,1,width).float() / (height -1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, d_model, width, height):
        super(PositionalEncoding2D, self).__init__()
        
        self.d_model = d_model
        self.height = height
        self.width = width
        
        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        
        self.register_buffer("positional_encoding", pe)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        pos_enc = self.positional_encoding.expand(batch_size, -1, -1, -1)
        return pos_enc 

# class PositionAttention(nn.Module):
#     """ Position attention module"""
#     def __init__(self, in_channels, width=16, height=8):
#         super(PositionAttention, self).__init__()
#         self.in_channels = in_channels

#         self.positional_encoding = PositionalEncoding2D(d_model=self.in_channels, width=16, height=8)

#         self.Q_conv = nn.Conv2d(self.in_channels, self.in_channels//4, kernel_size=1, stride=1, padding=0)
#         self.K_conv = nn.Conv2d(self.in_channels, self.in_channels//4, kernel_size=1, stride=1, padding=0)
#         self.V_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)

#         self.softmax = nn.Softmax(dim=-1)
#         self.scaling_factor = torch.sqrt(torch.tensor(in_channels)) 

#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         num_batch, num_channels, height, width = x.shape

#         pos_enc = self.positional_encoding(x)

#         x_pos_enc = x + pos_enc

#         q_val = self.Q_conv(x_pos_enc).view(num_batch, -1, height*width ).permute(0, 2, 1) # (B, HxW, C/8)
#         k_val = self.K_conv(x_pos_enc).view(num_batch, -1, height*width)                   # (B, C/8, HxW)
#         v_val = self.V_conv(x_pos_enc).view(num_batch, -1, height*width)                   # (B, C, HxW)

#         ### Scale dot-product attention
#         energy = torch.bmm(q_val, k_val)

#         ### Attention map (B, (HxW), (HxW))
#         attention = self.softmax(energy)

#         out = torch.bmm(v_val, attention.permute(0, 2, 1))
#         out = out.view(num_batch, num_channels, height, width)
#         return out + x_pos_enc, attention


class PositionAttention(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PositionAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out, attention