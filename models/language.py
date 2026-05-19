# ==============================================================================
# language.py — BERT Language Encoder cho TransVG
# ==============================================================================
# Wrapper quanh HuggingFace BertModel:
#
#   NestedTensor(word_ids [B, 17], attention_mask [B, 17])
#     → BERT (12 encoder layers)
#     → Lấy output layer thứ enc_num
#     → NestedTensor(text_features [B, 17, 768], mask [B, 17])
#
# So sánh với SeqTR:
#   SeqTR:   GloVe (300d) → BiGRU → max-pool → 1 vector [B, 1, 1024]
#   TransVG: BERT → 17 token vectors [B, 17, 768]
#
# Thay đổi so với repo gốc:
#   Gốc:  pytorch_pretrained_bert.modeling.BertModel (cũ, khó cài)
#   Mới:  transformers.BertModel (HuggingFace, ổn định, có sẵn trên Kaggle)
#
#   API khác nhau:
#     Gốc:  all_encoder_layers, _ = bert(ids, token_type_ids, mask)
#            output = all_encoder_layers[enc_num - 1]
#     Mới:  outputs = bert(ids, attention_mask=mask, output_hidden_states=True)
#            output = outputs.hidden_states[enc_num]
# ==============================================================================

import torch
from torch import nn

from transformers import BertModel
from utils.misc import NestedTensor


class BERTEncoder(nn.Module):
    """
    BERT language encoder cho TransVG.

    Nhận NestedTensor chứa token IDs + attention mask,
    trả NestedTensor chứa text features + padding mask.

    Attributes:
        num_channels (int): Kích thước output embedding (768 cho bert-base)
        enc_num (int): Dùng output của encoder layer thứ mấy (1-12)
    """

    def __init__(self, bert_model='bert-base-uncased', train_bert=True, enc_num=12):
        """
        Args:
            bert_model (str): Tên model trên HuggingFace
            train_bert (bool): Có fine-tune BERT không (lr_bert > 0)
            enc_num (int): Dùng output layer thứ mấy (12 = layer cuối = mạnh nhất)
                - 12: dùng full BERT (mặc định, kết quả tốt nhất)
                - 6: dùng nửa BERT (nhanh hơn, yếu hơn)
                - 0: chỉ dùng word embedding (không qua encoder)
        """
        super().__init__()

        # bert-base-uncased: 12 layers, hidden_size=768
        # bert-large-uncased: 24 layers, hidden_size=1024
        if 'base' in bert_model:
            self.num_channels = 768
        else:
            self.num_channels = 1024

        self.enc_num = enc_num

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model)

        # Freeze BERT nếu không fine-tune
        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, text_data: NestedTensor):
        """
        Args:
            text_data: NestedTensor chứa:
                - tensors: [B, seq_len] — BERT token IDs
                  Ví dụ: [101, 1996, 2158, 1999, 2417, 102, 0, 0, ...]
                          [CLS]  the   man   in   red  [SEP] PAD PAD
                - mask: [B, seq_len] — attention mask
                  1 = token thật (attend), 0 = padding (ignore)

        Returns:
            NestedTensor chứa:
                - tensors: [B, seq_len, 768] — text features
                - mask: [B, seq_len] — ĐÃ ĐẢO: True = padding, False = attend
                  (Đảo để khớp với convention key_padding_mask của Transformer)
        """
        input_ids = text_data.tensors       # [B, 17]
        attention_mask = text_data.mask      # [B, 17] — 1=attend, 0=padding

        if self.enc_num > 0:
            # Forward qua BERT, lấy hidden states của TẤT CẢ layers
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # hidden_states là tuple 13 tensors:
            #   [0] = word embeddings (trước encoder)
            #   [1] = output encoder layer 1
            #   ...
            #   [12] = output encoder layer 12 = last_hidden_state
            xs = outputs.hidden_states[self.enc_num]  # [B, 17, 768]
        else:
            # Chỉ dùng word embeddings (không qua encoder layers)
            xs = self.bert.embeddings.word_embeddings(input_ids)  # [B, 17, 768]

        # Đảo mask: 1→False (attend), 0→True (padding/ignore)
        # Để khớp với convention key_padding_mask: True = ignore
        mask = attention_mask.to(torch.bool)
        mask = ~mask  # Đảo: True = padding

        return NestedTensor(xs, mask)


def build_bert_encoder(config):
    """Factory function tạo BERTEncoder."""
    train_bert = config.lr_bert > 0
    return BERTEncoder(
        bert_model=config.bert_model,
        train_bert=train_bert,
        enc_num=config.bert_enc_num
    )


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config

    print("=== Test BERT Encoder ===\n")

    model = BERTEncoder(
        bert_model='bert-base-uncased',
        train_bert=True,
        enc_num=12
    )
    print(f"num_channels: {model.num_channels}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Test forward
    B = 2
    word_ids = torch.tensor([
        [101, 1996, 2158, 1999, 2417, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [101, 2187, 3899, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    word_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    text_data = NestedTensor(word_ids, word_mask)
    output = model(text_data)

    text_src, text_mask = output.decompose()
    print(f"\ntext_src shape:  {text_src.shape}")   # [2, 17, 768]
    print(f"text_mask shape: {text_mask.shape}")     # [2, 17]
    print(f"text_mask[0]:    {text_mask[0]}")         # False...False True...True
    print(f"text_mask dtype: {text_mask.dtype}")      # bool

    print("\n✅ BERT Encoder test passed!")
