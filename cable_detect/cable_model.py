# cable_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class CableLinkModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, pos_dim=32, visual_dim=128,
                 transformer_dim=256, num_heads=8, num_layers=3, mlp_dim=128):
        super().__init__()
        
        # 1. 视觉编码器 (使用预训练ResNet18的前几层)
        resnet = models.resnet18(pretrained=True)
        self.visual_encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            nn.AdaptiveAvgPool2d((1, 1)) # 输出 (B*N, visual_dim, 1, 1)
        )
        # ResNet18 layer2 a 128 a.
        self.visual_projector = nn.Linear(128, visual_dim)

        # 2. 标签编码器
        self.label_encoder = nn.Embedding(vocab_size, embed_dim)
        
        # 3. 位置编码器
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, pos_dim // 2),
            nn.ReLU(),
            nn.Linear(pos_dim // 2, pos_dim)
        )
        
        # 4. 特征融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(visual_dim + embed_dim + pos_dim, transformer_dim),
            nn.ReLU(),
            nn.LayerNorm(transformer_dim)
        )
        
        # 5. Transformer 上下文编码器
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 2,
            batch_first=True # (B, N, D)
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        # 6. 连接预测器 (MLP)
        self.link_predictor = nn.Sequential(
            nn.Linear(transformer_dim * 2, mlp_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, crops, pos, labels, mask):
        """
        crops: (B, N, C, H, W)
        pos: (B, N, 2)
        labels: (B, N)
        mask: (B, N) - True为有效
        """
        B, N, C, H, W = crops.shape
        
        # 1. 视觉特征
        vis_feat = self.visual_encoder(crops.view(B * N, C, H, W))
        vis_feat = vis_feat.view(B * N, -1) # (B*N, 128)
        vis_feat = self.visual_projector(vis_feat) # (B*N, visual_dim)
        vis_feat = vis_feat.view(B, N, -1) # (B, N, visual_dim)
        
        # 2. 标签特征
        lbl_feat = self.label_encoder(labels) # (B, N, embed_dim)
        
        # 3. 位置特征
        pos_feat = self.pos_encoder(pos) # (B, N, pos_dim)
        
        # 4. 融合
        fused_feat = torch.cat([vis_feat, lbl_feat, pos_feat], dim=-1)
        fused_feat = self.fusion_mlp(fused_feat) # (B, N, transformer_dim)
        
        # 应用掩码 (Transformer 期望 'True' 表示被遮盖)
        padding_mask = (mask == 0) # (B, N)
        
        # 5. Transformer
        ctx_feat = self.transformer(fused_feat, src_key_padding_mask=padding_mask) # (B, N, D)
        
        # 6. 成对预测
        feat_i = ctx_feat.unsqueeze(2).repeat(1, 1, N, 1) # (B, N, N, D)
        feat_j = ctx_feat.unsqueeze(1).repeat(1, N, 1, 1) # (B, N, N, D)
        
        pairs = torch.cat([feat_i, feat_j], dim=-1) # (B, N, N, 2*D)
        
        logits = self.link_predictor(pairs).squeeze(-1) # (B, N, N)
        
        # 确保对称性
        logits = (logits + logits.transpose(1, 2)) / 2
        
        # 掩码无效的配对 (例如, 节点与填充节点, 或填充与填充)
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) # (B, N, N)
        logits = logits.masked_fill(pair_mask == 0, -1e9) # 用极大负数填充
        
        return logits