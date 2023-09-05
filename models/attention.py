import torch
import torch.nn as nn
import math 

class LinearPositionalEncoding(nn.Module):
    def __init__(self, width=16, height=8):
        super(LinearPositionalEncoding, self).__init__()
        height_map = torch.arange(height).view(1, height, 1).repeat(1, 1, width) / (height - 1)
        width_map = torch.arange(width).view(1, 1, width).repeat(1, height, 1) / (width - 1)

        self.register_buffer('height_map', height_map, persistent=False)
        self.register_buffer('width_map', width_map, persistent=False)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        height_map = self.height_map.expand(batch_size, -1, -1, -1)
        width_map = self.width_map.expand(batch_size, -1, -1, -1)
        return torch.cat([height_map, width_map, x], dim=1)


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

class PositionAttention(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, width=16, height=8):
        super(PositionAttention, self).__init__()
        self.in_channels = in_channels
        # self.in_channels = in_channels + 2  ##([self.height_map, self.width_map, x])

        # self.positional_encoding = PositionalEncoding2D(d_model=self.in_channels, width=16, height=8)
        self.positional_encoding = LinearPositionalEncoding(width=16, height=8)

        # self.Q_conv = nn.Conv2d(self.in_channels+2, self.in_channels//8, kernel_size=1, stride=1, padding=0)
        # self.K_conv = nn.Conv2d(self.in_channels+2, self.in_channels//8, kernel_size=1, stride=1, padding=0)
        # self.V_conv = nn.Conv2d(self.in_channels+2, self.in_channels, kernel_size=1, stride=1, padding=0)

        self.Q_conv = nn.Conv2d(self.in_channels, self.in_channels//8, kernel_size=1, stride=1, padding=0)
        self.K_conv = nn.Conv2d(self.in_channels, self.in_channels//8, kernel_size=1, stride=1, padding=0)
        self.V_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Sequential(#nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1, bias=False),
                                   nn.LayerNorm([self.in_channels, height, width]),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self.gamma = nn.Parameter(torch.zeros(1))

        # self.scaling_factor = torch.sqrt(torch.tensor(self.in_channels)) 

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        num_batch, num_channels, height, width = x.shape

        # pos_enc = self.positional_encoding(x)
        # x_pos_enc = x + pos_enc

        # x_pos_enc = self.positional_encoding(x)

        x_pos_enc = x
        
        q_val = self.Q_conv(x_pos_enc).view(num_batch, -1, height*width ).permute(0, 2, 1) # (B, HxW, C/8)
        k_val = self.K_conv(x_pos_enc).view(num_batch, -1, height*width)                   # (B, C/8, HxW)
        v_val = self.V_conv(x_pos_enc).view(num_batch, -1, height*width)                   # (B, C, HxW)

        ### Scale dot-product attention
        energy = torch.bmm(q_val, k_val)

        ### Attention map (B, (HxW), (HxW))
        attention = self.softmax(energy)

        out = torch.bmm(v_val, attention.permute(0, 2, 1))
        out = out.view(num_batch, num_channels, height, width)
        out = out + x
        out = self.conv(out)

        return out, attention


# class PositionAttention(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PositionAttention, self).__init__()
#         self.chanel_in = in_dim

#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(in_dim),
#                                    nn.ReLU(),
#                                    nn.Dropout2d(0.1, False))
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)

#         # out = self.gamma*out + x
#         out = out + x

#         # print(self.gamma)
#         out = self.conv(out)

#         return out, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size need to be div by heads"

        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, self.embed_size, bias=False)

    def forward(self, x, mask=None):
        """
            inputs :
                x : input feature maps( N X C X H X W)
            returns :
                out : attention value + input feature
                attention: N X (HxW) X (HxW)
        """
        N, C, height, width = x.size()
        x_reshape = x.view(N, C, height*width).permute(0, 2, 1)  ### (N, HxW, C)

        query_len, key_len, value_len = x_reshape.shape[1], x_reshape.shape[1], x_reshape.shape[1] ### Sequence length, number of words

        queries = self.queries(x_reshape)  # (N, query_len, embed_size)
        keys = self.keys(x_reshape)  # (N, key_len, embed_size)
        values = self.values(x_reshape)  # (N, value_len, embed_size)

        ### Split embedding into number of HEADS pieces
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)

        ### Scale dot-product attention
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        # queries shape: (N, query_len, heads, head_dim) Moi mot word co n_heads moi head co head_dim
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)  Ma tran tuong quan giua cac word trong input (head x seq_len x seq_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / self.embed_size**(1/2), dim=-1)
        # print(torch.argmax(attention, dim=-1))
        out = torch.einsum('nhqk, nvhd->nqhd', [attention, values])
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)

        ### Concat
        out = out.reshape(N, query_len, self.heads*self.head_dim)

        out = self.fc_out(out)
        out = out.permute(0, 2, 1).view(N, C, height, width)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return out + x, attention