# ==============================================================================
# model.py — TransVG Model Tổng Hợp
# ==============================================================================
# Ghép 4 module lại thành 1 model hoàn chỉnh:
#
#   ┌─────────────┐    ┌─────────────┐
#   │  Ảnh (img)  │    │  Text (ids) │
#   └──────┬──────┘    └──────┬──────┘
#          ↓                  ↓
#   ┌─────────────┐    ┌─────────────┐
#   │  Backbone   │    │    BERT     │
#   │(ResNet+DETR)│    │ (12 layers) │
#   │→ 400 tokens │    │→ 17 tokens  │
#   └──────┬──────┘    └──────┬──────┘
#          ↓                  ↓
#     visu_proj           text_proj
#    (256→256)           (768→256)
#          ↓                  ↓
#          └───── + [REG] ────┘
#                    ↓
#            Concat: 418 tokens
#          + Positional Embedding
#                    ↓
#          ┌─────────────────┐
#          │  VL Transformer │
#          │   (6 layers)    │
#          └────────┬────────┘
#                   ↓
#          Output [REG] token
#                   ↓
#          ┌─────────────────┐
#          │    MLP Head     │
#          │  256→256→256→4  │
#          └────────┬────────┘
#                   ↓
#            sigmoid → bbox
#         [x_c, y_c, w, h] ∈ [0,1]
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_visual_encoder
from .language import build_bert_encoder
from .vl_transformer import build_vl_transformer


class MLP(nn.Module):
    """
    Multi-Layer Perceptron đơn giản.

    Cấu trúc:
        Linear(256, 256) → ReLU →
        Linear(256, 256) → ReLU →
        Linear(256, 4)           → output (KHÔNG có activation)

    Tại sao 3 layers?
        - 1 layer: quá đơn giản, underfitting
        - 2 layers: chưa đủ capacity
        - 3 layers: balance giữa capacity và overfitting
        - Nhiều hơn: không cải thiện nhiều cho 4 outputs
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # Tạo list các layer dimensions
        # [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        # Ví dụ: [256, 256, 256, 4]
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # ReLU cho tất cả layers trừ layer cuối
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransVG(nn.Module):
    """
    TransVG — Model hoàn chỉnh cho Visual Grounding.

    Input:
        - img_data:  NestedTensor(img [B,3,640,640], mask [B,640,640])
        - text_data: NestedTensor(word_ids [B,17], word_mask [B,17])

    Output:
        - pred_box: Tensor [B, 4] — normalized (x_c, y_c, w, h) ∈ [0,1]

    Pipeline:
        1. Visual: img → ResNet → DETR Encoder → visu_proj → 400 visual tokens
        2. Text:   ids → BERT → text_proj → 17 text tokens
        3. Concat:  [REG] + text + visual → 418 tokens
        4. VL Transformer: self-attention fusion (6 layers)
        5. Lấy [REG] output → MLP → sigmoid → bbox
    """

    def __init__(self, config):
        super().__init__()

        hidden_dim = config.vl_hidden_dim  # 256

        # Tính số tokens
        divisor = 16 if config.dilation else 32
        self.num_visu_token = int((config.imsize / divisor) ** 2)  # 400
        self.num_text_token = config.max_query_len + 2             # 17

        # 1. Visual Encoder (ResNet + DETR)
        self.visumodel = build_visual_encoder(config)

        # 2. Language Encoder (BERT)
        self.textmodel = build_bert_encoder(config)

        # 3. Projection layers — align dimensions
        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)  # 256→256
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)  # 768→256

        # 4. Special tokens
        num_total = 1 + self.num_text_token + self.num_visu_token  # 418
        self.reg_token = nn.Embedding(1, hidden_dim)               # [REG] learnable
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)    # Positional

        # 5. VL Transformer (fusion)
        self.vl_transformer = build_vl_transformer(config)

        # 6. MLP Head → bbox
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, img_data, text_data):
        """
        Forward pass.

        Args:
            img_data:  NestedTensor(img [B,3,640,640], mask [B,640,640])
            text_data: NestedTensor(word_ids [B,17], word_mask [B,17])

        Returns:
            pred_box: Tensor [B, 4] — (x_c, y_c, w, h) normalized ∈ [0,1]
        """
        bs = img_data.tensors.shape[0]

        # =====================================================================
        # Step 1: Visual backbone → 400 tokens
        # =====================================================================
        visu_mask, visu_src = self.visumodel(img_data)
        # visu_mask: [B, 400] — True = padding
        # visu_src:  [400, B, 256]

        visu_src = self.visu_proj(visu_src)  # [400, B, 256] (project nếu cần)

        # =====================================================================
        # Step 2: BERT → 17 tokens
        # =====================================================================
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        # text_src:  [B, 17, 768]
        # text_mask: [B, 17] — True = padding (đã đảo trong language.py)

        text_src = self.text_proj(text_src)   # [B, 17, 256]
        text_src = text_src.permute(1, 0, 2)  # [17, B, 256] (LenxBxC)
        text_mask = text_mask.flatten(1)       # [B, 17]

        # =====================================================================
        # Step 3: [REG] token → 1 token
        # =====================================================================
        # reg_token.weight: [1, 256] → expand thành [1, B, 256]
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)  # [1, B, 256]
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)  # [B, 1]

        # =====================================================================
        # Step 4: Concat → 418 tokens
        # =====================================================================
        # Thứ tự: [REG](1) + text(17) + visual(400) = 418
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)   # [418, B, 256]
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)  # [B, 418]
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [418, B, 256]

        # =====================================================================
        # Step 5: VL Transformer → lấy [REG] output
        # =====================================================================
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # [418, B, 256]
        vg_hs = vg_hs[0]  # Lấy position 0 = [REG] token → [B, 256]

        # =====================================================================
        # Step 6: MLP → sigmoid → bbox
        # =====================================================================
        pred_box = self.bbox_embed(vg_hs).sigmoid()  # [B, 4]
        # Output: (x_c, y_c, w, h) ∈ [0, 1]

        return pred_box


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config
    from utils.misc import NestedTensor

    print("=== Test TransVG Full Model ===\n")

    model = TransVG(Config)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")
    print(f"num_visu_token:   {model.num_visu_token}")
    print(f"num_text_token:   {model.num_text_token}")

    # Test forward
    B = 2
    img = torch.randn(B, 3, 640, 640)
    img_mask = torch.zeros(B, 640, 640, dtype=torch.bool)
    img_data = NestedTensor(img, img_mask)

    word_ids = torch.randint(100, 30000, (B, 17))
    word_mask = torch.ones(B, 17, dtype=torch.long)
    word_mask[0, 8:] = 0  # Câu 1: 8 tokens thật
    word_mask[1, 5:] = 0  # Câu 2: 5 tokens thật
    text_data = NestedTensor(word_ids, word_mask)

    model.eval()
    with torch.no_grad():
        pred_box = model(img_data, text_data)

    print(f"\npred_box shape: {pred_box.shape}")  # [2, 4]
    print(f"pred_box:       {pred_box}")           # values ∈ [0, 1]
    print(f"pred_box range: [{pred_box.min():.4f}, {pred_box.max():.4f}]")

    print("\n✅ TransVG full model test passed!")
