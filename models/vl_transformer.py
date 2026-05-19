# ==============================================================================
# vl_transformer.py — Vision-Language Transformer Encoder
# ==============================================================================
# Fusion module — kết hợp visual + text qua self-attention:
#
#   [REG] token  (1, B, 256)    ← learnable embedding, dùng để lấy output
#   Text tokens  (17, B, 256)   ← từ BERT, đã project
#   Visual tokens (400, B, 256) ← từ backbone, đã project
#   ─────────────────────────────
#   Concat → (418, B, 256)
#   + Positional Embedding (418, 256)
#   → VL Transformer Encoder (6 layers self-attention)
#   → Output tại position 0 ([REG]): (B, 256)
#
# Tại sao cần module riêng?
#   - DETR Encoder (trong backbone.py): self-attention CHỈ trên visual tokens
#   - VL Transformer: self-attention trên CẢ visual + text + [REG]
#     → [REG] token attend vào mọi token → tổng hợp thông tin cross-modal
#     → Chính output của [REG] sẽ được dùng để dự đoán bbox
#
# So với SeqTR:
#   SeqTR dùng tanh gating (nhân element-wise) → fusion đơn giản
#   TransVG dùng self-attention → mạnh hơn nhưng tốn GPU hơn
# ==============================================================================

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# ==============================================================================
# TRANSFORMER ENCODER LAYER (giống backbone.py, duplicate cho self-contained)
# ==============================================================================

class VLTransformerEncoderLayer(nn.Module):
    """
    1 layer trong VL Transformer Encoder.

    Giống TransformerEncoderLayer trong backbone.py:
        - Q, K = src + pos (positional encoding)
        - V = src (không cộng pos)
        - Post-Norm (mặc định)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class VLTransformerEncoder(nn.Module):
    """Stack N layers VLTransformerEncoderLayer."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, pos=None):
        """
        Args:
            src: [L, B, C] — concat tokens (L = 418 = 1+17+400)
            src_key_padding_mask: [B, L] — True = ignore
            pos: [L, B, C] — positional embedding

        Returns:
            output: [L, B, C]
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


# ==============================================================================
# WRAPPER CLASS
# ==============================================================================

class VLTransformer(nn.Module):
    """
    Vision-Language Transformer Encoder.

    Wrapper đơn giản quanh VLTransformerEncoder
    với xavier initialization.
    """

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()

        encoder_layer = VLTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation="relu", normalize_before=normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = VLTransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        """
        Args:
            src: [L, B, C] — concat [REG] + text + visual tokens
            mask: [B, L] — padding mask (True = ignore)
            pos_embed: [L, B, C] — positional embedding

        Returns:
            output: [L, B, C] — output tokens (lấy [0] cho [REG])
        """
        return self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)


def build_vl_transformer(config):
    """Factory function tạo VLTransformer."""
    return VLTransformer(
        d_model=config.vl_hidden_dim,
        nhead=config.vl_nheads,
        num_encoder_layers=config.vl_enc_layers,
        dim_feedforward=config.vl_dim_feedforward,
        dropout=config.vl_dropout,
        normalize_before=False,
    )


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test VL Transformer ===\n")

    model = VLTransformer(d_model=256, nhead=8, num_encoder_layers=6)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    # Test forward: 418 tokens = 1 (REG) + 17 (text) + 400 (visual)
    B = 2
    L = 418
    src = torch.randn(L, B, 256)
    mask = torch.zeros(B, L, dtype=torch.bool)
    # Giả lập: text padding ở vị trí 7-17 (token 1-17)
    mask[0, 7:18] = True
    mask[1, 5:18] = True
    pos = torch.randn(L, B, 256)

    output = model(src, mask, pos)
    print(f"Input shape:  {src.shape}")      # [418, 2, 256]
    print(f"Output shape: {output.shape}")   # [418, 2, 256]

    # Lấy [REG] token output
    reg_output = output[0]  # [B, 256]
    print(f"[REG] output: {reg_output.shape}")  # [2, 256]

    print("\n✅ VL Transformer test passed!")
