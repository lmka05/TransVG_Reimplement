# ==============================================================================
# backbone.py — Visual Encoder cho TransVG
# ==============================================================================
# Gom 4 file gốc (backbone, position_encoding, transformer, detr) thành 1:
#
#   NestedTensor(img, mask)
#     → ResNet-50 (FrozenBatchNorm2d, freeze layer0+1)
#     → Conv2d 1×1 (2048 → 256)
#     → + Sine Positional Encoding 2D
#     → DETR Transformer Encoder (6 layers, encoder-only)
#     → Output: visu_mask [B, 400], visu_src [400, B, 256]
# ==============================================================================

import copy
import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import NestedTensor


# ==============================================================================
# PHẦN 1: FROZEN BATCHNORM
# ==============================================================================

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d với parameters CỐ ĐỊNH (không update khi training).

    Tại sao cần?
        - ResNet pretrained trên ImageNet → BN statistics (mean, var) đã rất tốt
        - RefCOCO nhỏ hơn ImageNet rất nhiều → nếu update BN trên RefCOCO
          thì statistics sẽ bị nhiễu → kết quả kém hơn
        - Giải pháp: Freeze BN = giữ nguyên mean/var từ ImageNet

    So với nn.BatchNorm2d:
        - nn.BatchNorm2d: weight, bias là parameters (trainable)
                          running_mean, running_var là buffers (update khi train)
        - FrozenBatchNorm2d: TẤT CẢ đều là buffers (không update)
    """

    def __init__(self, n):
        super().__init__()
        # Đăng ký tất cả dưới dạng buffer (không phải parameter)
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        # Loại bỏ num_batches_tracked (có trong BN thường nhưng không có ở đây)
        key = prefix + 'num_batches_tracked'
        if key in state_dict:
            del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# ==============================================================================
# PHẦN 2: RESNET-50 BACKBONE
# ==============================================================================

class Backbone(nn.Module):
    """
    ResNet-50 backbone với FrozenBatchNorm2d.

    Freeze strategy:
        - layer0 (conv1 + bn1): FROZEN (low-level features, không cần train)
        - layer1: FROZEN
        - layer2, layer3, layer4: TRAINABLE (fine-tune cho visual grounding)

    Output: feature map từ layer4 → [B, 2048, 20, 20] (640/32=20)
    """

    def __init__(self, name='resnet50', dilation=False):
        super().__init__()

        # Tạo ResNet-50 với FrozenBatchNorm (KHÔNG load pretrained ở đây)
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=None,
            norm_layer=FrozenBatchNorm2d
        )

        # Freeze layer0 + layer1 (chỉ train layer2, 3, 4)
        for name_param, parameter in backbone.named_parameters():
            if 'layer2' not in name_param and 'layer3' not in name_param and 'layer4' not in name_param:
                parameter.requires_grad_(False)

        # Chỉ lấy output từ layer4
        return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = 2048

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor(img [B,3,640,640], mask [B,640,640])

        Returns:
            dict: {'0': NestedTensor(features [B,2048,20,20], mask [B,20,20])}
        """
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # Downsample mask cho khớp với feature map size
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


# ==============================================================================
# PHẦN 3: SINE POSITIONAL ENCODING 2D
# ==============================================================================

class PositionEmbeddingSine(nn.Module):
    """
    Mã hóa vị trí 2D bằng sin/cos — giống "Attention is All You Need" nhưng mở rộng cho 2D.

    Mỗi vị trí (y, x) trên feature map 20×20 được mã hóa thành 1 vector 256D:
        - 128 chiều đầu: mã hóa vị trí y (hàng) bằng sin/cos
        - 128 chiều sau: mã hóa vị trí x (cột) bằng sin/cos

    Tại sao sin/cos?
        - Không có learnable parameters → generalize tốt hơn
        - Tần số khác nhau giúp model phân biệt vị trí ở nhiều scale
        - normalize=True: tọa độ được normalize về [0, 2π] để không phụ thuộc kích thước ảnh
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128 (= hidden_dim / 2)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor(features [B,C,H,W], mask [B,H,W])

        Returns:
            pos: [B, 256, H, W] — positional encoding cho mỗi pixel
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        # not_mask: True ở vùng ảnh thật, False ở padding
        not_mask = ~mask

        # Cumsum → tạo tọa độ liên tục (1, 2, 3, ... cho mỗi hàng/cột)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, H, W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [B, H, W]

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Tạo các tần số khác nhau
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Áp dụng sin/cos
        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, 128]
        pos_y = y_embed[:, :, :, None] / dim_t  # [B, H, W, 128]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # Concat y + x → [B, H, W, 256] → permute → [B, 256, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ==============================================================================
# PHẦN 4: DETR TRANSFORMER ENCODER
# ==============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    1 layer trong DETR Transformer Encoder.

    Cấu trúc (Post-Norm, mặc định):
        src → Self-Attention(Q=src+pos, K=src+pos, V=src) → Add & LayerNorm
            → FFN(Linear→ReLU→Dropout→Linear) → Add & LayerNorm → output

    Điểm khác biệt so với Transformer chuẩn:
        Positional encoding được CỘNG vào Q và K nhưng KHÔNG cộng vào V.
        → Vị trí ảnh hưởng đến attention weights (qua Q, K)
        → Nhưng không ảnh hưởng đến giá trị được aggregate (V)
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
            # Pre-Norm: LayerNorm trước attention
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-Norm (mặc định): attention trước LayerNorm
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    """Stack N layers TransformerEncoderLayer."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


# ==============================================================================
# PHẦN 5: VISUAL ENCODER (WRAPPER GOM TẤT CẢ)
# ==============================================================================

class VisualEncoder(nn.Module):
    """
    Visual Encoder hoàn chỉnh cho TransVG:
        ResNet-50 → Conv1×1 → DETR Transformer Encoder

    Input:  NestedTensor(img [B,3,640,640], mask [B,640,640])
    Output: visu_mask [B, 400], visu_src [400, B, 256]

    num_channels = 256 (output dimension, = hidden_dim)
    """

    def __init__(self, config):
        super().__init__()

        # 1. ResNet-50 backbone
        self.backbone = Backbone(name=config.backbone, dilation=config.dilation)

        # 2. Positional encoding
        N_steps = config.hidden_dim // 2  # 128
        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

        # 3. DETR Transformer Encoder (nếu detr_enc_num > 0)
        if config.detr_enc_num > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.nheads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                normalize_before=config.pre_norm
            )
            encoder_norm = nn.LayerNorm(config.hidden_dim) if config.pre_norm else None
            self.transformer = TransformerEncoder(
                encoder_layer, config.detr_enc_num, encoder_norm
            )
            self.input_proj = nn.Conv2d(
                self.backbone.num_channels, config.hidden_dim, kernel_size=1
            )
        else:
            self.transformer = None

        # Output channels
        if self.transformer is not None:
            self.num_channels = config.hidden_dim  # 256
        else:
            self.num_channels = self.backbone.num_channels  # 2048

        self._reset_transformer_parameters()

    def _reset_transformer_parameters(self):
        """Xavier init cho transformer parameters."""
        if self.transformer is not None:
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, img_data: NestedTensor):
        """
        Args:
            img_data: NestedTensor(img [B,3,640,640], mask [B,640,640])

        Returns:
            visu_mask: Tensor [B, H*W] — True = padding, False = attend
            visu_src:  Tensor [H*W, B, C] — visual features
                       Nếu có DETR encoder: C = 256, enriched by self-attention
                       Nếu không: C = 2048, raw ResNet features
        """
        # Bước 1: ResNet-50 → features + mask + positional encoding
        features = self.backbone(img_data)
        src_nested = list(features.values())[-1]  # Lấy output layer4
        src, mask = src_nested.decompose()         # [B,2048,20,20], [B,20,20]
        pos = self.position_embedding(src_nested)  # [B,256,20,20]

        assert mask is not None

        if self.transformer is not None:
            # Bước 2: Project channels: 2048 → 256
            src = self.input_proj(src)  # [B, 256, 20, 20]

            # Bước 3: Flatten spatial dims → sequence
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)    # [H*W, B, C] = [400, B, 256]
            pos = pos.flatten(2).permute(2, 0, 1)    # [400, B, 256]
            mask = mask.flatten(1)                     # [B, 400]

            # Bước 4: DETR Encoder
            memory = self.transformer(src, src_key_padding_mask=mask, pos=pos)
            # memory: [400, B, 256]

            return mask, memory
        else:
            # Không có DETR encoder → trả raw features
            mask = mask.flatten(1)                     # [B, 400]
            src = src.flatten(2).permute(2, 0, 1)      # [400, B, 2048]
            return mask, src


def build_visual_encoder(config):
    """Factory function tạo VisualEncoder."""
    return VisualEncoder(config)


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config

    print("=== Test Visual Encoder ===\n")

    # Build model
    model = VisualEncoder(Config)
    print(f"Output channels: {model.num_channels}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")

    # Test forward
    B = 2
    img = torch.randn(B, 3, 640, 640)
    mask = torch.zeros(B, 640, 640, dtype=torch.bool)
    # Giả lập: ảnh thứ 2 có padding bên phải
    mask[1, :, 500:] = True

    img_data = NestedTensor(img, mask)
    visu_mask, visu_src = model(img_data)

    print(f"\nvisu_mask shape: {visu_mask.shape}")  # [2, 400]
    print(f"visu_src shape:  {visu_src.shape}")     # [400, 2, 256]
    print(f"visu_mask dtype: {visu_mask.dtype}")     # bool

    print("\n✅ Visual Encoder test passed!")
