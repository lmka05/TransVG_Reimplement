# ==============================================================================
# dataloader.py — DataLoader cho TransVG
# ==============================================================================
# File này xử lý việc GOM nhiều samples thành 1 BATCH:
#
#   Dataset trả: (img, mask, word_id, word_mask, bbox) cho 1 sample
#        ↓
#   DataLoader gom B samples → collate_fn gom thành:
#        ↓
#   (NestedTensor_img, NestedTensor_text, bbox_batch)
#
# So sánh với SeqTR:
#   SeqTR:   (imgs, ref_inds, gt_bboxes, img_shapes)  — 4 tensors đơn thuần
#   TransVG: (NestedTensor_img, NestedTensor_text, bbox)  — 2 NestedTensor + 1 tensor
#
# Tại sao dùng NestedTensor?
#   Transformer cần biết vùng nào là padding để ignore trong attention.
#   NestedTensor gói (tensor + mask) thành 1 object → code gọn hơn.
# ==============================================================================

from torch.utils.data import DataLoader
from utils.misc import collate_fn


def build_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    """
    Tạo DataLoader từ dataset.

    Args:
        dataset: RefCOCODataset instance
        batch_size (int): Số sample mỗi batch (ví dụ: 8)
        shuffle (bool): Xáo trộn data?
            - True cho train (mỗi epoch thứ tự khác nhau)
            - False cho val/test (thứ tự cố định)
        num_workers (int): Số process song song load data
            - Kaggle nên dùng 2 (đủ nhanh, không tốn quá nhiều RAM)

    Returns:
        DataLoader: Iterator trả batch dạng
            (NestedTensor_img, NestedTensor_text, bbox)

    Cách dùng:
        loader = build_dataloader(train_dataset, batch_size=8, shuffle=True)
        for img_data, text_data, bbox in loader:
            # img_data.tensors: [B, 3, 640, 640]
            # img_data.mask:    [B, 640, 640]
            # text_data.tensors: [B, 17]
            # text_data.mask:    [B, 17]
            # bbox:              [B, 4]
            ...
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,       # Từ utils/misc.py — gom thành NestedTensor
        pin_memory=True,             # Tăng tốc chuyển data CPU → GPU
        drop_last=(shuffle is True), # Bỏ batch cuối nếu thiếu (chỉ khi train)
    )


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config
    from datasets.dataset import RefCOCODataset

    print("=" * 60)
    print("TEST DATALOADER")
    print("=" * 60)

    # Tạo dataset
    val_dataset = RefCOCODataset(
        ann_file=Config.ann_file,
        img_dir=Config.img_dir,
        split='val',
        config=Config
    )

    # Tạo DataLoader
    val_loader = build_dataloader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0  # 0 cho test (tránh multiprocess issues)
    )

    # Lấy 1 batch
    print("\n--- Getting 1 batch ---")
    img_data, text_data, bbox = next(iter(val_loader))

    # Decompose NestedTensor
    imgs, img_masks = img_data.decompose()
    word_ids, word_masks = text_data.decompose()

    print(f"imgs shape:       {imgs.shape}")         # [4, 3, 640, 640]
    print(f"img_masks shape:  {img_masks.shape}")     # [4, 640, 640]
    print(f"word_ids shape:   {word_ids.shape}")      # [4, 17]
    print(f"word_masks shape: {word_masks.shape}")     # [4, 17]
    print(f"bbox shape:       {bbox.shape}")           # [4, 4]
    print(f"bbox sample:      {bbox[0]}")              # normalized xywh

    # Kiểm tra NestedTensor .to(device)
    import torch
    if torch.cuda.is_available():
        img_data_gpu = img_data.to('cuda')
        print(f"\nGPU tensors: {img_data_gpu.tensors.device}")
    else:
        print("\n(No CUDA available, skip GPU test)")

    print("\n✅ DataLoader test passed!")
