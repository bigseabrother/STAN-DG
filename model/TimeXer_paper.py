import pickle
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention
from layers.Embed import DataEmbedding, DataEmbedding_inverted, PositionalEmbedding
import numpy as np

from layers.Transformer_EncDec import Decoder, DecoderLayer
# airformer


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size)).to(
            device)  # mask for causality

    def forward(self, x):
        self.mask= self.mask.to(x.device)



        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)  # create local windows
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]

        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:  # reshape to the original size
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 dim,  # hidden dim
                 depth,  # the number of MSA in CT-MSA
                 heads,  # the number of heads
                 window_size,  # the size of local window
                 mlp_dim,  # mlp dimension
                 num_time,  # the number of time slot
                 dropout=0.,  # dropout rate
                 device=None):  # device, e.g., cuda
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim,
                                  heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)  # [b*n, t, c]
        x = x + self.pos_embedding  # [b*n, t, c]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)

        return x
class SpatialAttention(nn.Module):
    # dartboard project + MSA
    def __init__(self,
                 dim,
                 heads=4,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 num_sectors=25,
                 assignment=None,
                 mask=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_sector = num_sectors
        self.assignment = assignment  # [n, n, num_sector]
        self.mask = mask  # [n, num_sector]

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [b, n, c]

        B, N, C = x.shape
        self.assignment= self.assignment.to(x.device)
        self.mask= self.mask.to(x.device)

        # query: [bn, 1, c]
        # key/value target: [bn, num_sector, c]
        # [b, n, num_sector, c]
        pre_kv = torch.einsum('bnc,mnr->bmrc', x, self.assignment)

        pre_kv = pre_kv.reshape(-1, self.num_sector, C)  # [bn, num_sector, c]
        pre_q = x.reshape(-1, 1, C)  # [bn, 1, c]

        q = self.q_linear(pre_q).reshape(B*N, -1, self.num_heads, C //
                                         self.num_heads).permute(0, 2, 1, 3)  # [bn, num_heads, 1, c//num_heads]
        kv = self.kv_linear(pre_kv).reshape(B*N, -1, 2, self.num_heads,
                                            C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [bn, num_heads, num_sector, c//num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.reshape(B, N, self.num_heads, 1,
                            self.num_sector) + self.relative_bias # you can fuse external factors here as well
        mask = self.mask.reshape(1, N, 1, 1, self.num_sector)
        # masking
        attn = attn.masked_fill_(mask, float(
            "-inf")).reshape(B * N, self.num_heads, 1, self.num_sector).softmax(dim=-1)
        attn_score=attn.clone()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn_score

class DS_MSA(nn.Module):
    # Dartboard Spatial MSA
    def __init__(self,
                 dim,  # hidden dimension
                 depth,  # number of MSA in DS-MSA
                 heads,  # number of heads
                 mlp_dim,  # mlp dimension
                 assignment1,  # dartboard assignment matrix
                 mask1,  # mask
                assignment2,  # dartboard assignment matrix
                 mask2,  # mask
                 dropout=0.):  # dropout rate
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if assignment2==None:
                self.layers.append(nn.ModuleList([
                    SpatialAttention(dim, heads=heads, dropout=dropout,
                                    assignment=assignment1, mask=mask1,
                                    num_sectors=assignment1.shape[-1]),
                                    #assignment=assignment, mask=mask,
                                    #  num_sectors=assignment.shape[-1]),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

                ]))

            else:
                self.layers.append(nn.ModuleList([
                    SpatialAttention(dim, heads=heads, dropout=dropout,
                                    assignment=assignment1, mask=mask1,
                                    num_sectors=assignment1.shape[-1]),
                                    #assignment=assignment, mask=mask,
                                    #  num_sectors=assignment.shape[-1]),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    SpatialAttention(dim, heads=heads, dropout=dropout,
                                    assignment=assignment2, mask=mask2,
                                    num_sectors=assignment2.shape[-1]),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),

                ]))

    def forward(self, x):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)  # [b*t, n, c]
        # x = x + self.pos_embedding  # [b*t, n, c]  we use relative PE instead
        attention_weights = None
        for layer in self.layers:
            if len(layer)==4:
                attn1, ff1,attn2, ff2 =layer
            
                tmp_x,attn_weight=attn1(x) 
                x = tmp_x + x
                x = ff1(x) + x
                tmp_x,attn_weight2=attn2(x) 
                x = tmp_x + x
                x = ff2(x) + x
                attention_weights=torch.cat([attn_weight,attn_weight2], dim=-1)
            else:
                attn1, ff1=layer
            
                tmp_x,attn_weight=attn1(x) 
                x = tmp_x + x
                x = ff1(x) + x

            # attention_weights.append(attn_weight)

        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x, attention_weights
    
# Pre Normalization in Transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FFN in Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ============ 图

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # x = torch.einsum('nwl,vw->nvl',(x,A))(batch,n,dim)(n,n)
        # print(x.shape,A.shape)
        x = torch.einsum('nwl,nvw->nvl', (x, A))  # 批量传播
        return x.contiguous()


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout=0.2,alpha=0.1):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        # normalization to adj
        # d = adj.sum(1)
        # a = adj / d.view(-1, 1)



        # 邻接矩阵归一化（逐样本处理）
        d = adj.sum(dim=2, keepdim=True)  # (B, N, 1)
        d = torch.where(d == 0, torch.ones_like(d), d)  # 避免除零
        a = adj / d  # (B, N, N)
        h = x
        out = [h]
        for _ in range(self.gdep):
            # h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            h = F.dropout(h, self.dropout)
            h = self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=-1)
        ho = self.mlp(ho)
        return ho


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=1, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes

        self.lin1 = nn.Linear(dim,dim)
        self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, node_emb):
        B_N,  D = node_emb.shape
        
        # 时间维度聚合（选择最新时间步）
        # node_emb = x  # [B*N, D]
        
        # 转为批次形式 [B, N, D]
        batch_size = B_N // self.nnodes
        node_emb = node_emb.view(batch_size, self.nnodes, D)
        batch_size = node_emb.size(0)
        # print(node_emb.shape)
        nodevec1 = F.gelu(self.alpha*self.lin1(node_emb))
        nodevec2 = F.gelu(self.alpha*self.lin2(node_emb))
        # adj = F.relu(torch.mm(nodevec1, nodevec2.transpose(1,0)) - torch.mm(nodevec2, nodevec1.transpose(1,0)))
                # 批量计算非对称邻接矩阵
        adj = torch.einsum('bnd,bmd->bnm', nodevec1, nodevec2)  # (B, N, N)
        adj -= torch.einsum('bmd,bnd->bnm', nodevec2, nodevec1) # 方向性差异
        adj = F.relu(adj)  # (B, N, N)
        if self.k < node_emb.shape[1]:
            # n_nodes = node_emb.shape[0]
            # mask = torch.zeros(n_nodes, n_nodes).to(node_emb.device)
            # mask.fill_(float('0'))
            # s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
            # mask.scatter_(1,t1,s1.fill_(1))
            # adj = adj*mask
            mask = torch.zeros_like(adj)  # (B, N, N)
            for b in range(batch_size):
                # 添加微小扰动避免平局
                perturbed_adj = adj[b] + torch.rand_like(adj[b]) * 0.01
                # 取每行topk
                topk_val, topk_idx = torch.topk(perturbed_adj, self.k, dim=1)
                # 生成稀疏掩码
                mask[b].scatter_(1, topk_idx, torch.ones_like(topk_val))
            adj = adj * mask  # (B, N, N)
        return adj

# ===============
class MultiHeadGraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, n_heads=8, alpha=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.nnodes=nnodes
        # 多头投影层
        self.q_proj = nn.Linear(dim, dim * n_heads)
        self.k_proj = nn.Linear(dim, dim * n_heads)
        
        # 动态稀疏化参数
        self.threshold = nn.Parameter(torch.Tensor([0.])) # 初始值为0

        self.dropout = nn.Dropout(0.3)
            

        
    def forward(self, node_emb):
        # node_emb: (B, N, D)
        # B, N, D = node_emb.shape
        B_N,  D = node_emb.shape

        batch_size = B_N // self.nnodes
        node_emb = node_emb.view(batch_size, self.nnodes, D)
        B, N, D = node_emb.shape
    
        # 多头投影
        q = self.q_proj(node_emb).view(B, N, self.n_heads, D)  # (B, N, H, D)
        k = self.k_proj(node_emb).view(B, N, self.n_heads, D)  # (B, N, H, D)
        
        # 多头注意力得分
        attn_logits = torch.einsum('bnhd,bmhd->bhnm', q, k) / (D**0.5)  # (B, H, N, N)

        # 动态稀疏化
        adj =self._sparsify(attn_logits)  # (B, H, N, N)
        # print(torch.any(adj != 0).item())
        # print( self.threshold )
        # 多头融合
        return adj.mean(dim=1)  # (B, N, N)

    def _sparsify(self, attn_logits):
        # 自适应阈值过滤
        mask = (attn_logits > self.threshold).float()
        mask = (attn_logits > 0).float()

        adj = F.softmax(attn_logits, dim=-1) * mask
        
        # Top-k增强
        if self.k < adj.size(-1):
            topk_val, topk_idx = torch.topk(adj, self.k, dim=-1)
            adj = torch.zeros_like(adj).scatter(-1, topk_idx, topk_val)
            # print('top-k:'+str(self.k))
        
        return adj

class SpatioTemporalGraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, temp_dim=1024,alpha=1.0):
        super().__init__()
        # 空间关系学习
        self.spatial_constructor = MultiHeadGraphConstructor(nnodes, k, dim)
        
        # 时间动态学习
        self.temp_encoder = nn.GRU(dim, temp_dim, batch_first=True)
        self.temp_proj = nn.Linear(temp_dim, dim)
        self.basic_encoder = nn.GRU(dim, temp_dim, batch_first=True)
        self.basic_proj = nn.Linear(temp_dim, dim)
        
        # 自适应融合权重
        self.lambda_weight = nn.Parameter(torch.tensor(0.5)) 
    def forward(self, node_emb_seq,cross):
        # node_emb_seq: (B, T, N, D)
        # B, T, N, D = node_emb_seq.shape
        B_N,T, D = node_emb_seq.shape
        
        # 时间动态编码
        temp_feat, _ = self.temp_encoder(cross)
        temp_feat = self.temp_proj(temp_feat[:, -1, :])#.view(B, 1027, D)  # (B, N, D)
        basic_feat, _ = self.basic_encoder(node_emb_seq)
        basic_feat = self.basic_proj(basic_feat[:, -1, :])#.view(B, 1027, D)  # (B, N, D)
       
                # 自适应加权融合
        lambda_ = torch.sigmoid(self.lambda_weight)
        fused_emb = lambda_ * basic_feat + (1 - lambda_) * temp_feat
        # 生成动态图
        return self.spatial_constructor(fused_emb)
class AdaptiveMixprop(nn.Module):
    def __init__(self, c_in, c_out, max_depth=6, dropout=0.2,alpha=0.1):
        super().__init__()
        self.max_depth = max_depth
        self.dropout = dropout
        
        # 深度权重生成
        self.depth_weights = nn.Sequential(
            nn.Linear(c_in, 32),
            nn.ReLU(),
            nn.Linear(32, max_depth+1)
        )
        
        # 基础传播层
        self.convs = nn.ModuleList([
            nconv() for _ in range(max_depth)
        ])
        
        # 特征融合
        self.mlp = nn.Linear((max_depth+1)*c_in, c_out)

    def forward(self, x, adj):
        # x: (B, N, D)
        B, N, D = x.shape
        
        # 生成深度权重
        weights = F.softmax(self.depth_weights(x.mean(1)), dim=-1)  # (B, max_depth+1)
        # print("weights shape:", weights.shape)
        # print("h shape:", x.shape)   
        # 多阶传播
        h = x
        outputs = [weights[:, 0:1] * h]
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, adj)
            outputs.append(weights[:, i+1:i+2] * h)
        
        # 动态加权融合
        ho = torch.cat(outputs, dim=-1)
        return self.mlp(ho)

class BiResGCN(nn.Module):
    def __init__(self, c_in, c_out, n_layers=3):
        super().__init__()
        self.forward_conv = nn.ModuleList([
            nconv() for _ in range(n_layers)
        ])
        self.backward_conv = nn.ModuleList([
            nconv() for _ in range(n_layers)
        ])
        self.gate = nn.Linear(2*c_out, c_out)
        
    def forward(self, x, adj):
        # 正向传播
        h_fwd = x
        for conv in self.forward_conv:
            h_fwd = conv(h_fwd, adj)
        
        # 反向传播（转置邻接矩阵）
        h_bwd = x
        adj_t = adj.transpose(1,2)
        for conv in self.backward_conv:
            h_bwd = conv(h_bwd, adj_t)
        
        # 门控融合
        combined = torch.cat([h_fwd, h_bwd], dim=-1)
        return torch.sigmoid(self.gate(combined)) * h_fwd + (1 - torch.sigmoid(self.gate(combined))) * h_bwd
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1] #torch.Size([2054, 1, 64])
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # x shape: torch.Size([2054, 1, 4, 512])
        # glb shape: torch.Size([2054, 115, 1, 512])
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class EnEmbedding2(nn.Module):
    def __init__(self, stocks,nvars, d_model, patch_len, dropout):
        super(EnEmbedding2, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len*nvars, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, stocks, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)
        self.stocknums=stocks

    def forward(self, x):
        # do patching
        # (x_mark_enc[:b*stocks,vars,step]
        n_vars =  self.stocknums # x.shape[1] #torch.Size([2054, 1, 64])

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) #[:b*stocks,vars,numpatch,patchlen]
        x=x.permute(0,2,1,3)#[:b*stocks,numpatch,vars,patchlen]
        x = torch.reshape(x, (x.shape[0] , x.shape[1], x.shape[2]*x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) ##[:b*stocks,numpatch,dim]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))##[:b,stocks,numpatch,dim]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # x shape: torch.Size([2054, 1, 4, 512])
        # glb shape: torch.Size([2054, 115, 1, 512])
        x = torch.cat([x, glb], dim=2)##[:b,stocks,numpatch+cls,dim]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars

class EnEmbedding2_nopatch(nn.Module):
    def __init__(self, stocks,nvars, d_model, patch_len, dropout):
        super(EnEmbedding2_nopatch, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(19, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, stocks, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.stocknums=stocks

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        # (x_mark_enc[:b*stocks,vars,step]
        n_vars = self.stocknums # x.shape[1] #torch.Size([2054, 1, 64])
        x=x.permute(0,2,1)

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) ##[:b*stocks,numpatch,dim]

        return self.dropout(x), n_vars
class EnEmbedding2_noglb(nn.Module):
    def __init__(self, stocks,nvars, d_model, patch_len, dropout):
        super(EnEmbedding2_noglb, self).__init__()

        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len*nvars, d_model, bias=False)
        # self.glb_token = nn.Parameter(torch.randn(1, stocks, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.stocknums=stocks

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        # (x_mark_enc[:b*stocks,vars,step]
        n_vars = self.stocknums # x.shape[1] #torch.Size([2054, 1, 64])

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) #[:b*stocks,vars,numpatch,patchlen]
        x=x.permute(0,2,1,3)#[:b*stocks,numpatch,vars,patchlen]
        x = torch.reshape(x, (x.shape[0] , x.shape[1], x.shape[2]*x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) ##[:b*stocks,numpatch,dim]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))##[:b,stocks,numpatch,dim]

        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars
class ExEmbedding2(nn.Module):
    def __init__(self, stocks,nvars, d_model, patch_len, dropout):
        super(ExEmbedding2, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.nvars = nvars


        self.value_embedding = nn.Linear(patch_len*nvars, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, stocks, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.stocknums=stocks

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        # (x_mark_enc[:b*stocks,vars,step]
        n_vars = self.stocknums # x.shape[1] #torch.Size([1027, 118, 64])

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) #[:b*stocks,vars,numpatch,patchlen]torch.Size([1027, 118, 4, 16])
        x=x.permute(0,2,1,3)#[:b*stocks,numpatch,vars,patchlen]
        x = torch.reshape(x, (x.shape[0] , x.shape[1], x.shape[2]*x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) ##[:b*stocks,numpatch,dim]
        x = torch.reshape(x, (-1,n_vars, x.shape[-2], x.shape[-1]))##[:b,stocks,numpatch,dim]
        # glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # x shape: torch.Size([2054, 1, 4, 512])
        # glb shape: torch.Size([2054, 115, 1, 512])
        # x = torch.cat([x, glb], dim=2)##[:b,stocks,numpatch+cls,dim]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars
    
    

class GraphEncoder(nn.Module):
    def __init__(self, layers, norm_layer, 
                 gl_layer,#node_embs,
                stocknums=1027,
                #  activation,conv1,conv2,dropout,
                 projection=None):
        super(GraphEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.stocknums=stocknums

        # add
        self.graph_learning = gl_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        adj = self.graph_learning(x,cross)#self.node_embs) 
        # 筛选出非零元素
        # print(torch.any(adj != 0))

        for i, layer in enumerate(self.layers):
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta,i=i,adj=adj,layernum=len(self.layers),node_embs=None,stocknums=self.stocknums)#self.node_embs)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x,adj

class  Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None,stocknums=1027):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.stocknums=stocknums

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        attention=[]
        for layer in self.layers:
            x,attention_weights= layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta,stocknums=self.stocknums)
            if attention_weights is not None:
                attention.append(attention_weights)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        attn=None
        if len(attention)>0:
            attn=torch.cat(attention, dim=0)
        return x,attn

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu",
                 ds_layer=None,ct_layer=None
                 ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.ds_layer = ds_layer
        self.ct_layer = ct_layer


    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None,stocknums=1027):
        attention_weights=None
        B, L, D = cross.shape #torch.Size([1027, 4, 512])
        if self.ds_layer!=None:
            cross,attention_weights = self.ds_layer(cross.reshape(-1,stocknums,L,D).permute(0,3,1,2)) #ds_MSA需要bcnt
            cross=cross.permute(0,2,3,1).reshape(B,L,D)


        # _, T, _ = cross.shape #torch.Size([1027, 4, 512])bn,t,c
        if self.ct_layer!=None:

            cross = self.ct_layer(cross.reshape(-1,stocknums,L,D).permute(0,3,1,2)) #ds_MSA需要bcnt

            cross=cross.permute(0,2,3,1).reshape(B,L,D)
        if x!=None:
            cross_attn = self.dropout(self.cross_attention(
                cross,x,x,# x_glb,x_glb,
                attn_mask=cross_mask,
                tau=tau, delta=delta
            )[0])
            # cross_attn = torch.reshape(cross_attn,
                                    #    (cross_attn.shape[0] * cross_attn.shape[1], cross_attn.shape[2])).unsqueeze(1)
            cross = cross + cross_attn
            cross = self.norm2(cross)



        
        # y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = cross#x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # return self.norm3(x + y)
        return self.norm3(y + cross),attention_weights

class EnhancedMixprop(mixprop):
    def __init__(self, c_in, c_out, gdep, dropout=0.2, alpha=0.1):
        super().__init__(c_in, c_out, gdep, dropout, alpha)
        # 财务特征门控机制
        self.finance_gate = nn.Sequential(
            nn.Linear(c_in, c_in),
            nn.Sigmoid()
        )
        
    def forward(self, x, adj, finance_feat=None):
        # 财务特征引导的门控
        if finance_feat is not None:
            gate = self.finance_gate(finance_feat)
            x = x * gate
        
        # 原始传播过程
        d = adj.sum(1)
        a = adj / d.view(-1, 1)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = F.dropout(h, self.dropout)
            h = self.nconv(h,a)
            # 添加财务特征残差连接
            if finance_feat is not None:
                h = h + 0.5 * finance_feat
            out.append(h)
        ho = torch.cat(out,dim=2)
        ho = self.mlp(ho)
        return ho
class GraphEncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, 
                 
                 
                 gnn_layers,   cls_len,graph_norm_layer,
                 
                 d_ff=None,
                 dropout=0.1, activation="relu",
                 ds_layer=None,
                 ct_layer=None,
                #  curlayer=0
                 ):
        super(GraphEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu


        self.graph_layers = nn.ModuleList(gnn_layers)
        # self.graph_learning = gl_layer
        self.graphnorm = graph_norm_layer
        self.cls_len = cls_len
        # self.node_embs = node_embs

        self.ds_layer = ds_layer
        self.ct_layer = ct_layer
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.gru2 = nn.GRU(d_model, d_model, batch_first=True)

        # 可学习的权重
        self.weight = nn.Parameter(torch.tensor(0.5))
        # self.bi_gcn = BiResGCN(d_model, d_model)
        # self.spatial_constructor = MultiHeadGraphConstructor(1027, 16, d_model)


    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None,i=0,adj=None,layernum=0,node_embs=None,stocknums=1027):
        # 空间注意力
        B, L, D = cross.shape #torch.Size([1027, 4, 512])bn,t,c
        if self.ds_layer!=None:
            cross,attention_weights = self.ds_layer(cross.reshape(-1,stocknums,L,D).permute(0,3,1,2)) #ds_MSA需要bcnt
            cross=cross.permute(0,2,3,1).reshape(B,L,D)

        # 时间注意力
        if self.ct_layer!=None:
            cross = self.ct_layer(cross.reshape(-1,stocknums,L,D).permute(0,3,1,2)) #ds_MSA需要bcnt
            cross=cross.permute(0,2,3,1).reshape(B,L,D)

        # 基本面自注意力
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x) #torch.Size([1027, 5, 512] 归一化

        gcls_len=1
        if i < len(self.graph_layers):
            g = x[:, -1, :]#.unsqueeze(1)
            # y经过GRU处理
            y_output, _ = self.gru(cross)
            y_last = y_output[:, -1]
            x_output, _ = self.gru2(x)
            g = x_output[:, -1]
            # 按照权重相加           
            lambda_ = torch.sigmoid(self.weight)

            fused_feat = lambda_ * y_last + (1 - lambda_) * g

            g = rearrange(fused_feat.unsqueeze(1), '(b n) p d -> (b p) n d', n=stocknums)
            g = self.graph_layers[i](g, adj) + g
            # adj=self.spatial_constructor(g.reshape(-1,g.shape[-1]),adj) 
            g = rearrange(g, '(b p) n d -> (b n) p d', p=gcls_len)
            x = torch.cat([x[:, :-1, :], g], dim=1) 

            x = self.graphnorm(x)

        if x!=None:
            # x_glb = torch.reshape(x_glb_ori, (B, -1, D))
            cross_attn = self.dropout(self.cross_attention(
                cross, x,x,#x_glb,x_glb,
                attn_mask=cross_mask,
                tau=tau, delta=delta
            )[0])
            # cross_attn = torch.reshape(cross_attn,
                                    #    (cross_attn.shape[0] * cross_attn.shape[1], cross_attn.shape[2])).unsqueeze(1)
            cross = cross + cross_attn
            cross = self.norm2(cross)
#
        y = cross#x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(y + cross)#x)

class Model(nn.Module):

    def __init__(self, configs,device):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        if  configs.patch_len==0:
            self.patch_num = 0
        else:
            self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        # self.n_vars =  configs.enc_in

        self.index_num=    configs.index_num
        self.stocknums= configs.stocknums
        # add
        cls_len = 1
        gdep = configs.graph_depth
        knn = configs.knn
        embed_dim = configs.embed_dim
        gc_alpha=1
        self.graph_flag=configs.graph_flag

        if configs.data=='stock0index' or configs.data=='stock45index':
            self.ex_embedding3 = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        else:
            if configs.data=='stock3geo':
                if self.patch_len==0:
                    self.en_embedding2 = EnEmbedding2_nopatch(self.stocknums,configs.index_num, configs.d_model, self.patch_len, configs.dropout) #4 9
                else:
                    if configs.index_num==115: #正常默认
                        self.en_embedding2 = EnEmbedding2_noglb(self.stocknums,19,  configs.d_model,self.patch_len, configs.dropout) #14财务特征+5市场化指数
                    elif configs.index_num==28: #行业地理加入图增强（消融）
                        self.en_embedding2 = EnEmbedding2_noglb(self.stocknums,24,  configs.d_model,self.patch_len, configs.dropout) #14财务特征+5市场化指数+3行业分类层级+2地域分类层级
                    else: # 其他
                        self.en_embedding2 = EnEmbedding2_noglb(self.stocknums,configs.index_num,  configs.d_model,self.patch_len, configs.dropout) #4 9

            else:
                self.en_embedding2 = EnEmbedding2(self.stocknums,9, configs.d_model, self.patch_len, configs.dropout) #4 9
            # self.ex_embedding2 = ExEmbedding2(1027,(configs.enc_in+4), configs.d_model, self.patch_len, configs.dropout)
            self.ex_embedding3 = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        
        self.get_dartboard_info(device,configs.ds_matrix)
        # print(device)
        if configs.graph_flag:

            if configs.ct_flag and configs.ds_flag:
                self.encoder = GraphEncoder(
                [
                    GraphEncoderLayer(
                        # 用于基本面自注意力
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        # 用于交叉注意力
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        # 用于图增强
                        gnn_layers=[AdaptiveMixprop(configs.d_model, configs.d_model, gdep) for _ in range(configs.e_layers-1)],

                        cls_len=1,#无用
                        #图增强后归一化
                        graph_norm_layer=torch.nn.LayerNorm(configs.d_model),

                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                        # 空间注意力
                        ds_layer=DS_MSA(configs.d_model,
                                                 depth=1,
                                                 heads=configs.n_heads,
                                                 mlp_dim=configs.d_ff,
                                                 assignment1=self.assignment1,
                                                 assignment2=self.assignment2,

                                                 mask1=self.mask1,
                                                 mask2=self.mask2,

                                                 dropout=configs.dropout),
                        # 时间注意力
                        ct_layer=CT_MSA(configs.d_model,
                                            depth=1,
                                            heads=configs.n_heads,
                                            window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                            mlp_dim=configs.d_ff,
                                            num_time=self.seq_len,
                                            device=device
                                            ),
                        
                    )
                    for l in range(configs.e_layers)
                ],
                # 归一化层
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                # 图学习层
                gl_layer= SpatioTemporalGraphConstructor(self.stocknums, knn, 512, alpha=gc_alpha),
                stocknums=self.stocknums

            )
            elif configs.ct_flag and not configs.ds_flag:
                self.encoder = GraphEncoder(
                [
                    GraphEncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,

                        gnn_layers=[AdaptiveMixprop(configs.d_model, configs.d_model, gdep) for _ in range(configs.e_layers-1)],
                        # gl_layer= graph_constructor(1027, knn, embed_dim, alpha=gc_alpha),
                        # node_embs=nn.Parameter(torch.randn(1027, embed_dim), requires_grad=True), 
                        cls_len=1,
                        graph_norm_layer=torch.nn.LayerNorm(configs.d_model),

                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,

                        ds_layer=None,#DS_MSA(configs.d_model,
                        #                          depth=1,
                        #                          heads=configs.n_heads,
                        #                          mlp_dim=configs.d_ff,
                        #                          assignment=self.assignment,
                        #                          mask=self.mask,
                        #                          dropout=configs.dropout),
                        ct_layer=CT_MSA(configs.d_model,
                                            depth=1,
                                            heads=configs.n_heads,
                                            window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                            mlp_dim=configs.d_ff,
                                            num_time=self.seq_len,
                                            device=device
                                            )
                    )
                    for l in range(configs.e_layers)
                ],

                norm_layer=torch.nn.LayerNorm(configs.d_model),
    
                gl_layer= SpatioTemporalGraphConstructor(self.stocknums, knn, 512, alpha=gc_alpha),
                # node_embs=nn.Parameter(torch.randn(1027, embed_dim), requires_grad=True), 
                stocknums=self.stocknums

                # activation=configs.activation,
                # conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1),
                # conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1),
                # dropout = nn.Dropout(configs.dropout)

            )
            elif not configs.ct_flag and  configs.ds_flag:
                self.encoder = GraphEncoder(
                [
                    GraphEncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,

                        gnn_layers=[AdaptiveMixprop(configs.d_model, configs.d_model, gdep) for _ in range(configs.e_layers-1)],
                        # gl_layer= graph_constructor(1027, knn, embed_dim, alpha=gc_alpha),
                        # node_embs=nn.Parameter(torch.randn(1027, embed_dim), requires_grad=True), 
                        cls_len=1,
                        graph_norm_layer=torch.nn.LayerNorm(configs.d_model),

                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,

                        ds_layer=DS_MSA(configs.d_model,
                                                 depth=1,
                                                 heads=configs.n_heads,
                                                 mlp_dim=configs.d_ff,
                                                 assignment1=self.assignment1,
                                                 assignment2=self.assignment2,

                                                 mask1=self.mask1,
                                                 mask2=self.mask2,
                                                #  assignment=self.assignment,
                                                #  mask=self.mask,
                                                 dropout=configs.dropout),
                        ct_layer=None,#CT_MSA(configs.d_model,
                                            # depth=1,
                                            # heads=configs.n_heads,
                                            # window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                            # mlp_dim=configs.d_ff,
                                            # num_time=self.seq_len,
                                            # device=device
                                            # )
                    )
                    for l in range(configs.e_layers)
                ],

                norm_layer=torch.nn.LayerNorm(configs.d_model),
    
                gl_layer= SpatioTemporalGraphConstructor(self.stocknums, knn, 512, alpha=gc_alpha),
                # node_embs=nn.Parameter(torch.randn(1027, embed_dim), requires_grad=True), 
                stocknums=self.stocknums

                # activation=configs.activation,
                # conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1),
                # conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1),
                # dropout = nn.Dropout(configs.dropout)

            )
            elif not configs.ct_flag and not configs.ds_flag:
                self.encoder = GraphEncoder(
                [
                    GraphEncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,

                        gnn_layers=[AdaptiveMixprop(configs.d_model, configs.d_model, gdep) for _ in range(configs.e_layers-1)],
                        # gl_layer= graph_constructor(1027, knn, embed_dim, alpha=gc_alpha),
                        # node_embs=nn.Parameter(torch.randn(1027, embed_dim), requires_grad=True), 
                        cls_len=1,
                        graph_norm_layer=torch.nn.LayerNorm(configs.d_model),

                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,

                        ds_layer=None,#DS_MSA(configs.d_model,
                                                #  depth=1,
                                                #  heads=configs.n_heads,
                                                #  mlp_dim=configs.d_ff,
                                                #  assignment=self.assignment,
                                                #  mask=self.mask,
                                                #  dropout=configs.dropout),
                        ct_layer=None,#CT_MSA(configs.d_model,
                                            # depth=1,
                                            # heads=configs.n_heads,
                                            # window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                            # mlp_dim=configs.d_ff,
                                            # num_time=self.seq_len,
                                            # device=device
                                            # )
                    )
                    for l in range(configs.e_layers)
                ],

                norm_layer=torch.nn.LayerNorm(configs.d_model),
    
                gl_layer= SpatioTemporalGraphConstructor(self.stocknums, knn, 512, alpha=gc_alpha),
                # node_embs=nn.Parameter(torch.randn(1027, embed_dim), requires_grad=True), 
                stocknums=self.stocknums

                # activation=configs.activation,
                # conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1),
                # conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1),
                # dropout = nn.Dropout(configs.dropout)

            )

        else:
            if configs.ct_flag and configs.ds_flag:
                self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                        ds_layer=DS_MSA(configs.d_model,
                                             depth=1,
                                             heads=configs.n_heads,
                                             mlp_dim=configs.d_ff,
                                            #  assignment=self.assignment,
                                            #  mask=self.mask,
                                            assignment1=self.assignment1,
                                            assignment2=self.assignment2,

                                            mask1=self.mask1,
                                            mask2=self.mask2,
                                             dropout=configs.dropout),
                        ct_layer=CT_MSA(configs.d_model,
                                         depth=1,
                                         heads=configs.n_heads,
                                         window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                         mlp_dim=configs.d_ff,
                                         num_time=self.seq_len,
                                           device=device
                                         )
                    )
                    for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                stocknums=self.stocknums
            )
            elif configs.ct_flag and not configs.ds_flag:
                self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                        ds_layer=None,#DS_MSA(configs.d_model,
                                            #  depth=1,
                                            #  heads=configs.n_heads,
                                            #  mlp_dim=configs.d_ff,
                                            #  assignment=self.assignment,
                                            #  mask=self.mask,
                                            #  dropout=configs.dropout),
                        ct_layer=CT_MSA(configs.d_model,
                                         depth=1,
                                         heads=configs.n_heads,
                                         window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                         mlp_dim=configs.d_ff,
                                         num_time=self.seq_len,
                                           device=device
                                         )
                    )
                    for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                stocknums=self.stocknums
            )        
            elif not configs.ct_flag and configs.ds_flag:
                self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                        ds_layer=DS_MSA(configs.d_model,
                                             depth=1,
                                             heads=configs.n_heads,
                                             mlp_dim=configs.d_ff,
                                            #  assignment=self.assignment,
                                            #  mask=self.mask,
                                                 assignment1=self.assignment1,
                                                 assignment2=self.assignment2,

                                                 mask1=self.mask1,
                                                 mask2=self.mask2,
                                             dropout=configs.dropout),
                        ct_layer=None,#CT_MSA(configs.d_model,
                                        #  depth=1,
                                        #  heads=configs.n_heads,
                                        #  window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                        #  mlp_dim=configs.d_ff,
                                        #  num_time=self.seq_len,
                                        #    device=device
                                        #  )
                    )
                    for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                stocknums=self.stocknums
            )  
            elif not configs.ct_flag and not configs.ds_flag:
                self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                        ds_layer=None,#DS_MSA(configs.d_model,
                                            #  depth=1,
                                            #  heads=configs.n_heads,
                                            #  mlp_dim=configs.d_ff,
                                            #  assignment=self.assignment,
                                            #  mask=self.mask,
                                            #  dropout=configs.dropout),
                        ct_layer=None,#CT_MSA(configs.d_model,
                                        #  depth=1,
                                        #  heads=configs.n_heads,
                                        #  window_size= configs.seq_len // 2 ** (configs.e_layers - l - 1),
                                        #  mlp_dim=configs.d_ff,
                                        #  num_time=self.seq_len,
                                        #    device=device
                                        #  )
                    )
                    for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                stocknums=self.stocknums
            )   
        self.head_nf = configs.d_model# * (self.patch_num + 1)
        # self.end_conv_1 = nn.Conv2d(self.head_nf, configs.enc_in * self.pred_len, kernel_size=(1, self.patch_num))#+1))
        self.end_conv_1 = nn.Conv2d(self.head_nf, configs.enc_in * self.pred_len, kernel_size=(1, self.seq_len))#+1))

    def get_dartboard_info(self,device,ds_matrix):
        '''
        get dartboard-related attributes
        '''
        # path_assignment = '/home/yc/geodata1209/data_output/assignment_code.npy'
        # path_mask = '/home/yc/geodata1209/data_output/mask_code.npy'
        
        '''
        距离
        '''
        if self.stocknums==204:
            pathname= '300'
            pathname2='300'
        elif self.stocknums==310:
            pathname='500'
            pathname2='500'

        elif self.stocknums==513:
            pathname='1000'
            pathname2='1000'

        else:
            pathname='32'
            pathname2='4'


        
        if ds_matrix==0 or ds_matrix==3 :
            path_assignment2 = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/assignment_region_'+pathname2+'.npy'
            path_mask2 = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/mask_region_'+pathname2+'.npy'
            # '''行业'''
            path_assignment = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/assignment_industry_'+pathname+'.npy'
            path_mask = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/mask_industry_'+pathname+'.npy'
        elif ds_matrix==1:
            path_assignment = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/assignment_industry_'+pathname+'.npy'
            path_mask = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/mask_industry_'+pathname+'.npy'
        elif ds_matrix==2:
            path_assignment2 = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/assignment_region_'+pathname2+'.npy'
            path_mask2 = '/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/mask_region_'+pathname2+'.npy'
        assignment1=None
        assignment2=None
        if ds_matrix==0 or ds_matrix==3 :

            assignment1=torch.from_numpy(np.load(path_assignment)).float()
            mask1=torch.from_numpy(np.load(path_mask)).float()

            assignment2=torch.from_numpy(np.load(path_assignment2)).float()
            mask2=torch.from_numpy(np.load(path_mask2)).float()
        elif ds_matrix==1:
            assignment1=torch.from_numpy(np.load(path_assignment)).float()
            mask1=torch.from_numpy(np.load(path_mask)).float()
        elif ds_matrix==2:
            assignment1=torch.from_numpy(np.load(path_assignment2)).float()
            mask1=torch.from_numpy(np.load(path_mask2)).float()
        if ds_matrix==3 :
            assignment1 = torch.cat((assignment1, assignment2), dim=2)
            mask1 = torch.cat((mask1, mask2), dim=1)
            assignment2=None
            mask2=None
        print(assignment1!=None,assignment2!=None)
        # 检查结果的形状
        # print(path_assignment,path_assignment2)

        # print(assignment.shape)
        # self.assignment = assignment.float().to(device)
        # self.mask = mask.bool().to(device)
        if assignment1!=None:
            self.assignment1 = assignment1.float().to(device)
            self.mask1 = mask1.bool().to(device)
        else:
            self.assignment1=None
            self.mask1=None

        if assignment2!=None:
            self.assignment2 = assignment2.float().to(device)
            self.mask2 = mask2.bool().to(device)
        else:
            self.assignment2=None
            self.mask2=None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if len(x_enc.shape)==4:
            b,  t , n,c= x_enc.shape
            x_enc = x_enc.permute(0,2,1, 3 ).reshape(b*n,t,  c)
            _,_,_,times= x_mark_enc.shape
            _,labellen,_,_ = x_dec.shape
            x_mark_enc = x_mark_enc.permute(0,2,1, 3 ).reshape(b*n,t,  times)
            _,labellen,_,_= x_mark_dec.shape #torch.Size([4, 40, 471, 4])
            x_dec = x_dec.permute(0,2,1, 3 ).reshape(b*n,labellen,  c)
            x_mark_dec = x_mark_dec.permute(0,2,1, 3 ).reshape(b*n,labellen,  times)

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape #torch.Size([2054, 64, 114])

        if  self.index_num!=4 and times==4:
            ex_embed2 = self.ex_embedding3(x_enc,x_mark_enc)#torch.Size([2054, numpatchs+1, 512])
            en_embed2=None
        else:
            if   self.index_num==14 or  self.index_num==5 or  self.index_num==4 :
                en_embed2, _ = self.en_embedding2(x_mark_enc[:,:,:].permute(0, 2, 1))#torch.Size([2054, numpatchs+1, 512])
            else:
                en_embed2, _ = self.en_embedding2(x_mark_enc[:,:,:-4].permute(0, 2, 1))#x_mark_enc[:,:,:-4] 只使用了基本面，没有使用4个时间特征
            ex_embed2 = self.ex_embedding3(x_enc,None)

        attn=None
        enc_out,attn = self.encoder(en_embed2, ex_embed2) #[bs x nvars x d_model x patch_num]

        enc_out = torch.reshape(
            enc_out, (-1, self.stocknums, enc_out.shape[-2], enc_out.shape[-1])) # b,n,ptach,c
        b,n,patch,c = enc_out.shape

        # # z: [bs x n x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2).view(b*n,c,1, patch) #torch.Size([1027, 512, 1, 5])
        dec_out = self.end_conv_1(enc_out).view(b*n, N, self.pred_len)
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)#torch.Size([1027, 24, 114])

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,attn


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1)) # patch
        ex_embed = self.ex_embedding(x_enc, x_mark_enc) # 按变量embedding
        # ex_embed_time = self.embedding_air(inputs[..., -4:].long())

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out,attn = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :],attn  # [B, L, D]
        else:
            return None