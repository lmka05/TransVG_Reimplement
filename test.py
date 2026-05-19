# ==============================================================================
# test.py — Test TransVG trên val/testA/testB
# ==============================================================================
# Chạy: python test.py
#
# Workflow:
#   1. Load config
#   2. Build model
#   3. Load best_checkpoint.pth
#   4. Đánh giá trên val, testA, testB
#   5. In bảng kết quả
#
# Output:
#   ========================================
#   TransVG Evaluation Results (RefCOCO)
#   ========================================
#   val   : Acc@0.5 = 78.23% | mIoU = 65.12%
#   testA : Acc@0.5 = 81.15% | mIoU = 68.34%
#   testB : Acc@0.5 = 72.89% | mIoU = 60.78%
#   ========================================
# ==============================================================================

import os
import sys
import time

import torch
from tqdm import tqdm

from config import Config
from models import build_model
from datasets import build_val_loader
from evaluate import trans_vg_eval_test
from utils.box_utils import xywh2xyxy, bbox_iou


@torch.no_grad()
def evaluate_split(model, data_loader, device):
    """
    Đánh giá model trên 1 split (val/testA/testB).

    Gom TẤT CẢ predictions trước → tính metrics 1 lần
    (chính xác hơn tính per-batch rồi average).

    Args:
        model: TransVG model (eval mode)
        data_loader: DataLoader cho split cần đánh giá
        device: 'cuda' hoặc 'cpu'

    Returns:
        accuracy: float — Acc@0.5 (0.0 → 1.0)
        mean_iou: float — mean IoU
    """
    model.eval()

    pred_box_list = []
    gt_box_list = []

    for batch in tqdm(data_loader, desc="Evaluating"):
        img_data, text_data, target = batch

        # Move to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # Forward
        pred_boxes = model(img_data, text_data)  # [B, 4]

        # Collect predictions
        pred_box_list.append(pred_boxes.cpu())
        gt_box_list.append(target.cpu())

    # Gom tất cả
    pred_boxes = torch.cat(pred_box_list, dim=0)  # [N, 4]
    gt_boxes = torch.cat(gt_box_list, dim=0)      # [N, 4]
    total_num = gt_boxes.shape[0]

    # Tính Acc@0.5
    accu_num = trans_vg_eval_test(pred_boxes, gt_boxes)
    accuracy = accu_num / total_num

    # Tính mean IoU
    pred_xyxy = xywh2xyxy(pred_boxes)
    gt_xyxy = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_xyxy, gt_xyxy)
    mean_iou = iou.mean().item()

    return accuracy, mean_iou


def main():
    config = Config
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # 1. Build model
    # =========================================================================
    print("\n--- Building model ---")
    model = build_model(config)
    model.to(device)

    # =========================================================================
    # 2. Load checkpoint
    # =========================================================================
    checkpoint_path = os.path.join(config.output_dir, 'best_checkpoint.pth')

    # Cho phép override bằng command line argument
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--checkpoint' and i + 1 < len(sys.argv):
                checkpoint_path = sys.argv[i + 1]

    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Hãy train model trước (python train.py) hoặc chỉ định checkpoint:")
        print("  python test.py --checkpoint /path/to/best_checkpoint.pth")
        sys.exit(1)

    print(f"\n--- Loading checkpoint: {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint.get('epoch', '?')
    best_accu = checkpoint.get('best_accu', '?')
    print(f"Checkpoint epoch: {epoch}, best_accu: {best_accu}")

    # =========================================================================
    # 3. Đánh giá trên từng split
    # =========================================================================
    splits = ['val', 'testA', 'testB']
    results = {}

    for split in splits:
        print(f"\n--- Evaluating on {split} ---")
        try:
            data_loader = build_val_loader(config, split=split)
            accuracy, mean_iou = evaluate_split(model, data_loader, device)
            results[split] = {'accu': accuracy, 'miou': mean_iou}
            print(f"  {split}: Acc@0.5 = {accuracy*100:.2f}% | mIoU = {mean_iou*100:.2f}%")
        except Exception as e:
            print(f"  {split}: SKIPPED ({e})")
            results[split] = {'accu': 0, 'miou': 0}

    # =========================================================================
    # 4. In bảng kết quả
    # =========================================================================
    print("\n" + "=" * 50)
    print("  TransVG Evaluation Results (RefCOCO)")
    print("=" * 50)
    for split in splits:
        r = results[split]
        print(f"  {split:6s}: Acc@0.5 = {r['accu']*100:6.2f}% | mIoU = {r['miou']*100:6.2f}%")
    print("=" * 50)

    # =========================================================================
    # 5. Save kết quả
    # =========================================================================
    import json
    results_path = os.path.join(config.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
