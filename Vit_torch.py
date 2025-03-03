import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, normal_

class AddPositionEmbs(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, emb_dim) * 0.02)
    
    def forward(self, x):
        return x + self.pos_embedding

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        xavier_uniform_(self.fc1.weight)
        normal_(self.fc1.bias, std=1e-6)
        xavier_uniform_(self.fc2.weight)
        normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

class Encoder1DBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, num_heads, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MlpBlock(hidden_dim, mlp_dim, dropout)
        
        # 注意力层初始化
        xavier_uniform_(self.attn.in_proj_weight)
        normal_(self.attn.in_proj_bias, std=1e-6)
        xavier_uniform_(self.attn.out_proj.weight)
        normal_(self.attn.out_proj.bias, std=1e-6)

    def forward(self, x):
        attn_output, _ = self.attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x)
        )
        x = x + self.dropout(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, mlp_dim, num_heads, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder1DBlock(hidden_dim, mlp_dim, num_heads, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.pos_emb = AddPositionEmbs(seq_len=197, emb_dim=hidden_dim)  # 默认ViT-B/16
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, train=True):
        x = self.pos_emb(x)
        x = self.dropout(x) if train else x
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class VisionTransformer(nn.Module):
    def __init__(self, 
                 num_classes, 
                 img_size=224,
                 patch_size=16,
                 hidden_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_dim=3072,
                 dropout=0.1,
                 attn_dropout=0.1,
                 representation_size=None):
        
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer编码器
        self.encoder = Encoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # 分类头
        self.pre_logits = nn.Identity()
        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(hidden_dim, representation_size),
                nn.Tanh()
            )
            hidden_dim = representation_size
            
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._init_weights()

    def _init_weights(self):
        # 卷积层初始化
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.normal_(self.patch_embed.bias, std=1e-6)
        
        # 分类头初始化
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        # 分块嵌入 [B, C, H, W] -> [B, hidden_dim, grid, grid]
        x = self.patch_embed(x)  
        B, C, H, W = x.shape
        
        # 展平并转置 [B, C, H*W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)  
        
        # 添加分类token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer编码
        x = self.encoder(x, self.training)
        
        # 分类
        x = x[:, 0]  # 取分类token
        x = self.pre_logits(x)
        return self.head(x)