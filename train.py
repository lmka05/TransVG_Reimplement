# ==============================================================================
# train.py — Training Script cho TransVG
# ==============================================================================
# Entry point chính — chạy: python train.py
#
# Pipeline:
#   1. Set seed → reproducible
#   2. Build model (TransVG)
#   3. Build DataLoader (train + val)
#   4. Setup optimizer (AdamW, 4 param groups)
#   5. Setup LR scheduler (StepLR, drop ở epoch 60)
#   6. Resume từ checkpoint (nếu có)
#   7. Training loop: train → validate → save checkpoint
# ==============================================================================

import os
import sys
import math
import time
import json
import random
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from models import build_model
from datasets import build_train_loader, build_val_loader
from evaluate import trans_vg_loss, trans_vg_eval_val
from utils.misc import MetricLogger, SmoothedValue


# ==============================================================================
# PHẦN 1: TRAIN 1 EPOCH
# ==============================================================================

def train_one_epoch(model, data_loader, optimizer, device, epoch, max_norm=0):
    """
    Train model trong 1 epoch.

    Args:
        model: TransVG model
        data_loader: train DataLoader
        optimizer: AdamW optimizer
        device: 'cuda' hoặc 'cpu'
        epoch: epoch hiện tại (để in log)
        max_norm: gradient clipping (0 = không clip)

    Returns:
        dict: {'loss': avg_loss, 'loss_bbox': ..., 'loss_giou': ..., 'lr': ...}
    """
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 50

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        img_data, text_data, target = batch

        # 1. Move to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # 2. Forward
        pred_box = model(img_data, text_data)  # [B, 4]

        # 3. Compute loss
        loss_dict = trans_vg_loss(pred_box, target)
        total_loss = loss_dict['loss_bbox'] + loss_dict['loss_giou']
        loss_value = total_loss.item()

        # 4. Check NaN/Inf
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # 5. Backward
        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # 6. Log
        metric_logger.update(loss=loss_value)
        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ==============================================================================
# PHẦN 2: VALIDATE
# ==============================================================================

@torch.no_grad()
def validate(model, data_loader, device):
    """
    Đánh giá model trên validation set.

    Returns:
        dict: {'miou': mean_iou, 'accu': accuracy_at_0.5}
    """
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 50, header):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)

        # Move to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # Forward
        pred_boxes = model(img_data, text_data)

        # Compute metrics
        miou, accu = trans_vg_eval_val(pred_boxes, target)

        # Update logger (weighted by batch_size)
        metric_logger.meters['miou'].update(torch.mean(miou).item(), batch_size)
        metric_logger.meters['accu'].update(accu, batch_size)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


# ==============================================================================
# PHẦN 3: MAIN
# ==============================================================================

def main():
    config = Config

    # =========================================================================
    # 1. Set seed
    # =========================================================================
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # 2. Build model
    # =========================================================================
    print("\n--- Building model ---")
    model = build_model(config)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable params: {n_parameters:,}")

    # =========================================================================
    # 3. Setup optimizer — 4 param groups với lr khác nhau
    # =========================================================================
    # Group 1: ResNet backbone → fine-tune nhẹ (pretrained ImageNet)
    visu_cnn_param = [p for n, p in model.named_parameters()
                      if "visumodel" in n and "backbone" in n and p.requires_grad]

    # Group 2: DETR Transformer Encoder → fine-tune nhẹ
    visu_tra_param = [p for n, p in model.named_parameters()
                      if "visumodel" in n and "backbone" not in n and p.requires_grad]

    # Group 3: BERT → fine-tune nhẹ
    text_tra_param = [p for n, p in model.named_parameters()
                      if "textmodel" in n and p.requires_grad]

    # Group 4: Còn lại (VL Transformer, projections, [REG], MLP) → train mạnh
    rest_param = [p for n, p in model.named_parameters()
                  if "visumodel" not in n and "textmodel" not in n and p.requires_grad]

    param_list = [
        {"params": rest_param,       "lr": config.lr},            # 1e-4
        {"params": visu_cnn_param,   "lr": config.lr_visu_cnn},   # 1e-5
        {"params": visu_tra_param,   "lr": config.lr_visu_tra},   # 1e-5
        {"params": text_tra_param,   "lr": config.lr_bert},       # 1e-5
    ]

    print(f"\nParam groups:")
    print(f"  rest (VL Trans + MLP):  {sum(p.numel() for p in rest_param):>10,} params, lr={config.lr}")
    print(f"  visu_cnn (ResNet):      {sum(p.numel() for p in visu_cnn_param):>10,} params, lr={config.lr_visu_cnn}")
    print(f"  visu_tra (DETR Enc):    {sum(p.numel() for p in visu_tra_param):>10,} params, lr={config.lr_visu_tra}")
    print(f"  text_tra (BERT):        {sum(p.numel() for p in text_tra_param):>10,} params, lr={config.lr_bert}")

    optimizer = torch.optim.AdamW(param_list, lr=config.lr, weight_decay=config.weight_decay)

    # =========================================================================
    # 4. Setup LR scheduler
    # =========================================================================
    if config.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    elif config.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / config.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif config.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / config.epochs) ** 0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")

    # =========================================================================
    # 5. Build data loaders
    # =========================================================================
    print("\n--- Building data loaders ---")
    train_loader = build_train_loader(config)
    val_loader = build_val_loader(config, split='val')

    # =========================================================================
    # 6. Resume from checkpoint (nếu có)
    # =========================================================================
    start_epoch = 0
    best_accu = 0

    if config.resume and os.path.isfile(config.resume):
        print(f"\n--- Resuming from {config.resume} ---")
        checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_accu' in checkpoint:
            best_accu = checkpoint['best_accu']
        print(f"Resumed: epoch={start_epoch}, best_accu={best_accu:.4f}")
    else:
        print("\n--- Training from scratch (ImageNet pretrained ResNet) ---")

    # =========================================================================
    # 7. Create output directory
    # =========================================================================
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with (output_dir / "config.txt").open("w") as f:
        for k, v in vars(config).items():
            if not k.startswith('_'):
                f.write(f"{k}: {v}\n")

    # =========================================================================
    # 8. Training loop
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Start training: {config.epochs} epochs, batch_size={config.batch_size}")
    print(f"Output dir: {config.output_dir}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(start_epoch, config.epochs):
        # --- Train ---
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config.clip_max_norm
        )

        # --- Update LR ---
        lr_scheduler.step()

        # --- Validate ---
        val_stats = validate(model, val_loader, device)

        # --- Log ---
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
        }

        # Print summary
        print(f"\n>>> Epoch [{epoch}/{config.epochs-1}]"
              f"  train_loss={train_stats.get('loss', 0):.4f}"
              f"  val_accu={val_stats.get('accu', 0):.4f}"
              f"  val_miou={val_stats.get('miou', 0):.4f}"
              f"  best={best_accu:.4f}\n")

        # Save log
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        # --- Save checkpoints ---
        # Always save latest checkpoint (for resume)
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_accu': best_accu,
        }
        torch.save(checkpoint_data, output_dir / 'checkpoint.pth')

        # Save best checkpoint
        val_accu = val_stats.get('accu', 0)
        if val_accu > best_accu:
            best_accu = val_accu
            checkpoint_data['best_accu'] = best_accu
            torch.save(checkpoint_data, output_dir / 'best_checkpoint.pth')
            print(f"  ★ New best accuracy: {best_accu:.4f}")

        # Save milestone checkpoints (every 10 epochs + before lr drop)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.lr_drop:
            torch.save(checkpoint_data, output_dir / f'checkpoint_epoch{epoch:03d}.pth')

    # =========================================================================
    # 9. Done
    # =========================================================================
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\nTraining complete!")
    print(f"Total time: {total_time_str}")
    print(f"Best val accuracy: {best_accu:.4f}")
    print(f"Checkpoints saved to: {config.output_dir}")


if __name__ == '__main__':
    main()
