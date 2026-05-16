# ==============================================================================
# datasets/__init__.py — Hàm tiện ích tạo DataLoader
# ==============================================================================
# Cung cấp 2 hàm chính để dùng trong train.py:
#
#   from datasets import build_train_loader, build_val_loader
#   train_loader = build_train_loader(config)       # 1 dòng là xong
#   val_loader = build_val_loader(config, 'val')     # 1 dòng là xong
# ==============================================================================

from .dataset import RefCOCODataset
from .dataloader import build_dataloader


def build_train_loader(config):
    """
    Tạo DataLoader cho tập TRAIN.

    Tự động:
        - Load annotations từ config.ann_file
        - Áp dụng train transforms (augmentation mạnh)
        - Shuffle = True, drop_last = True

    Args:
        config: Config object

    Returns:
        DataLoader — mỗi batch trả (NestedTensor_img, NestedTensor_text, bbox)
    """
    dataset = RefCOCODataset(
        ann_file=config.ann_file,
        img_dir=config.img_dir,
        split='train',
        config=config,
    )

    loader = build_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    return loader


def build_val_loader(config, split='val'):
    """
    Tạo DataLoader cho tập VAL/TEST.

    Tự động:
        - Load annotations từ config.ann_file
        - Áp dụng val transforms (chỉ resize + normalize, không augment)
        - Shuffle = False, drop_last = False

    Args:
        config: Config object
        split (str): 'val', 'testA', 'testB'

    Returns:
        DataLoader
    """
    dataset = RefCOCODataset(
        ann_file=config.ann_file,
        img_dir=config.img_dir,
        split=split,
        config=config,
    )

    loader = build_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return loader
