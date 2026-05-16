# ==============================================================================
# dataset.py — RefCOCO Dataset cho TransVG
# ==============================================================================
# File này xử lý 1 SAMPLE duy nhất:
#   Load ảnh → Load text → Load bbox → Transform → Return 5 tensors
#
# Luồng xử lý 1 sample:
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │ instances.json  ──→  annotation (image_id, bbox, expressions)       │
#   │        │                                                            │
#   │        ├── Load ảnh (PIL) ──┐                                       │
#   │        ├── Lấy text ────────┤                                       │
#   │        └── Lấy bbox ────────┤                                       │
#   │                             ↓                                       │
#   │              image_transforms (resize, crop, flip, pad, normalize)  │
#   │              ──→ img [3,640,640] + mask [640,640] + bbox [4]        │
#   │                             ↓                                       │
#   │              text_transforms (BERT tokenize)                        │
#   │              ──→ word_id [17] + word_mask [17]                      │
#   │                             ↓                                       │
#   │              Return: (img, mask, word_id, word_mask, bbox)          │
#   └──────────────────────────────────────────────────────────────────────┘
#
# So sánh với SeqTR:
#   SeqTR:   return (img, ref_inds, gt_bbox, img_shapes)   — 4 tensors
#   TransVG: return (img, mask, word_id, word_mask, bbox)   — 5 tensors
#
# Khác biệt chính:
#   - SeqTR không có mask (CNN backbone không cần)
#   - SeqTR tokenize bằng GloVe vocab → 1 tensor (ref_inds)
#   - TransVG tokenize bằng BERT → 2 tensor (word_id + word_mask)
#   - SeqTR bbox = pixel [x1,y1,x2,y2] → quantize thành bins
#   - TransVG bbox = normalized [x_c, y_c, w, h] ∈ [0,1]
# ==============================================================================

import os
import json
import random

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.image_transform import make_transforms
from utils.text_transform import TextTransform


class RefCOCODataset(Dataset):
    """
    Dataset class cho RefCOCO — dùng với PyTorch DataLoader.

    Mỗi sample gồm 5 thành phần:
        - img:       Tensor [3, 640, 640]  — ảnh đã normalize + pad
        - img_mask:  Tensor [640, 640]     — 0 = pixel thật, 1 = padding
        - word_id:   Tensor [17]           — BERT token IDs
        - word_mask: Tensor [17]           — 1 = token thật, 0 = padding
        - bbox:      Tensor [4]            — normalized [x_c, y_c, w, h] ∈ [0,1]

    Annotation format (instances.json — dùng chung với SeqTR):
        {
            "train": [
                {
                    "image_id": 139,
                    "bbox": [100, 50, 200, 150],       ← COCO format [x, y, w, h]
                    "expressions": ["the man in red", "person on left"]
                },
                ...
            ],
            "val": [...], "testA": [...], "testB": [...]
        }
    """

    def __init__(self, ann_file, img_dir, split, config):
        """
        Args:
            ann_file (str): Đường dẫn tới instances.json
            img_dir (str): Thư mục chứa ảnh COCO train2014
            split (str): 'train', 'val', 'testA', 'testB'
            config: Config object chứa hyperparameters
        """
        super().__init__()

        self.img_dir = img_dir
        self.split = split

        # 1. Load annotations cho split cụ thể
        with open(ann_file, 'r') as f:
            anns_all = json.load(f)
        self.anns = anns_all[split]

        # 2. Tạo image transform pipeline
        #    train: multi-scale resize + crop + flip + color jitter + normalize + pad
        #    val/test: resize(640) + normalize + pad
        self.img_transform = make_transforms(config, split)

        # 3. Tạo text transform (BERT tokenizer)
        self.text_transform = TextTransform(
            bert_model=config.bert_model,
            max_query_len=config.max_query_len
        )

        print(f"[{split}] Loaded {len(self.anns)} samples")

    def __len__(self):
        """Trả về tổng số samples."""
        return len(self.anns)

    def __getitem__(self, idx):
        """
        Lấy 1 sample tại vị trí idx.
        DataLoader sẽ gọi hàm này cho mỗi sample.

        Returns:
            img:       Tensor [3, 640, 640]
            img_mask:  Tensor [640, 640]
            word_id:   Tensor [17]
            word_mask: Tensor [17]
            bbox:      Tensor [4]
        """
        ann = self.anns[idx]

        # =====================================================================
        # Bước 1: Load ảnh (PIL Image)
        # =====================================================================
        img_path = os.path.join(
            self.img_dir,
            "COCO_train2014_%012d.jpg" % ann['image_id']
        )
        img = Image.open(img_path).convert('RGB')

        # =====================================================================
        # Bước 2: Lấy text expression
        # =====================================================================
        expressions = ann['expressions']
        if self.split == 'train':
            # Training: random chọn 1 câu → data augmentation cho text
            # Mỗi epoch, cùng 1 ảnh có thể đi kèm câu mô tả khác nhau
            expression = random.choice(expressions)
        else:
            # Val/Test: luôn lấy câu đầu tiên → kết quả consistent
            expression = expressions[0]

        expression = expression.lower()

        # =====================================================================
        # Bước 3: Chuyển bbox format
        # =====================================================================
        # Annotation gốc: COCO format [x, y, w, h]
        #   x, y = góc trên-trái
        #   w, h = chiều rộng, chiều cao
        #
        # Chuyển sang: [x1, y1, x2, y2] (corner format)
        #   x1, y1 = góc trên-trái
        #   x2, y2 = góc dưới-phải
        x, y, w, h = ann['bbox']
        bbox = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)

        # =====================================================================
        # Bước 4: Image transforms
        # =====================================================================
        # Gom img + box + text vào 1 dict → transform đồng bộ cả 3
        # (flip ảnh → flip bbox + swap "left"↔"right" trong text)
        input_dict = {
            'img': img,          # PIL Image
            'box': bbox,         # [x1, y1, x2, y2] pixel coords
            'text': expression   # raw text string
        }

        input_dict = self.img_transform(input_dict)

        # Sau transform:
        #   img:  Tensor [3, 640, 640] — normalized + padded
        #   mask: Tensor [640, 640]    — 0=ảnh thật, 1=padding
        #   box:  Tensor [4]           — normalized [x_c, y_c, w, h] ∈ [0,1]
        #   text: str (có thể đã swap left↔right nếu bị flip)
        img = input_dict['img']
        img_mask = input_dict['mask']
        bbox = input_dict['box']
        text_after_transform = input_dict['text']

        # =====================================================================
        # Bước 5: Text transforms (BERT tokenize)
        # =====================================================================
        # Dùng text SAU transform (vì flip có thể swap "left"↔"right")
        word_id, word_mask = self.text_transform(text_after_transform)

        # Chuyển thành tensor
        word_id = torch.tensor(word_id, dtype=torch.long)
        word_mask = torch.tensor(word_mask, dtype=torch.long)

        return img, img_mask, word_id, word_mask, bbox


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config

    print("=" * 60)
    print("TEST DATASET")
    print("=" * 60)

    # Tạo dataset
    print("\n--- Creating val dataset ---")
    val_dataset = RefCOCODataset(
        ann_file=Config.ann_file,
        img_dir=Config.img_dir,
        split='val',
        config=Config
    )

    # Lấy 1 sample
    print("\n--- Getting 1 sample ---")
    img, img_mask, word_id, word_mask, bbox = val_dataset[0]
    print(f"img shape:       {img.shape}")         # [3, 640, 640]
    print(f"img dtype:       {img.dtype}")          # float32
    print(f"img range:       [{img.min():.2f}, {img.max():.2f}]")
    print(f"img_mask shape:  {img_mask.shape}")     # [640, 640]
    print(f"img_mask unique: {img_mask.unique()}")  # [0, 1]
    print(f"word_id shape:   {word_id.shape}")      # [17]
    print(f"word_id:         {word_id}")
    print(f"word_mask shape: {word_mask.shape}")     # [17]
    print(f"word_mask:       {word_mask}")
    print(f"bbox shape:      {bbox.shape}")          # [4]
    print(f"bbox:            {bbox}")                # normalized xywh ∈ [0,1]

    # Decode text
    decoded = val_dataset.text_transform.decode(word_id.tolist())
    print(f"decoded text:    '{decoded}'")

    print("\n✅ Dataset test passed!")
