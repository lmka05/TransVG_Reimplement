# ==============================================================================
# evaluate.py — Đánh giá TransVG
# ==============================================================================
# Cung cấp 3 hàm chính:
#
#   1. trans_vg_loss()     — Tính L1 + GIoU loss (dùng khi training)
#   2. trans_vg_eval_val() — Tính IoU + Acc@0.5 cho 1 batch (dùng khi validate)
#   3. trans_vg_eval_test()— Tính tổng samples đúng (dùng khi test toàn bộ)
#
# So sánh với SeqTR:
#   SeqTR:   Cross-Entropy loss, dequantize bins → pixel bbox → IoU
#   TransVG: L1 + GIoU loss, normalized xywh → xyxy → IoU
# ==============================================================================

import torch
import torch.nn.functional as F

from utils.box_utils import xywh2xyxy, bbox_iou, generalized_box_iou


# ==============================================================================
# PHẦN 1: LOSS FUNCTION
# ==============================================================================

def trans_vg_loss(pred_boxes, target_boxes):
    """
    Tính loss cho TransVG = L1 Loss + GIoU Loss.

    Args:
        pred_boxes:   [B, 4] — model output, normalized (x_c, y_c, w, h) ∈ [0,1]
        target_boxes: [B, 4] — ground truth, normalized (x_c, y_c, w, h) ∈ [0,1]

    Returns:
        dict: {
            'loss_bbox': L1 loss (scalar),
            'loss_giou': GIoU loss (scalar)
        }

    Tại sao dùng 2 loss?
        - L1 loss: Phạt khoảng cách tuyệt đối giữa pred và target.
          Ưu: ổn định, dễ optimize. Nhược: không hiểu geometry (overlap).
        - GIoU loss: Phạt dựa trên overlap thực tế giữa 2 boxes.
          Ưu: hiểu spatial relationship. Nhược: gradient yếu khi boxes xa nhau.
        - Kết hợp cả 2 → bổ trợ lẫn nhau.

    Ví dụ:
        pred   = [0.5, 0.5, 0.3, 0.4]  → tâm (0.5, 0.5), rộng 0.3, cao 0.4
        target = [0.5, 0.5, 0.3, 0.4]  → trùng hoàn toàn
        → L1 = 0, GIoU = 0 (perfect)
    """
    batch_size = pred_boxes.shape[0]

    # --- L1 Loss ---
    # |pred - target| cho từng tọa độ, rồi cộng lại
    loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction='none')
    # [B, 4] → sum / B
    loss_bbox = loss_bbox.sum() / batch_size

    # --- GIoU Loss ---
    # Chuyển xywh → xyxy để tính GIoU
    pred_xyxy = xywh2xyxy(pred_boxes)      # [B, 4]
    target_xyxy = xywh2xyxy(target_boxes)  # [B, 4]

    # generalized_box_iou trả [B, B] matrix → lấy diagonal (so sánh 1-1)
    giou_matrix = generalized_box_iou(pred_xyxy, target_xyxy)  # [B, B]
    loss_giou = 1 - torch.diag(giou_matrix)  # [B]
    loss_giou = loss_giou.sum() / batch_size

    return {
        'loss_bbox': loss_bbox,
        'loss_giou': loss_giou,
    }


# ==============================================================================
# PHẦN 2: EVALUATION FUNCTIONS
# ==============================================================================

def trans_vg_eval_val(pred_boxes, target_boxes):
    """
    Đánh giá 1 batch — dùng trong validate() khi training.

    Args:
        pred_boxes:   [B, 4] — normalized (x_c, y_c, w, h)
        target_boxes: [B, 4] — normalized (x_c, y_c, w, h)

    Returns:
        miou: Tensor [B] — IoU cho từng sample
        accu: float — số sample có IoU ≥ 0.5

    Ví dụ:
        B = 4, IoU = [0.8, 0.3, 0.6, 0.9]
        → accu = 3 (3 samples có IoU ≥ 0.5)
    """
    # Chuyển xywh → xyxy
    pred_xyxy = xywh2xyxy(pred_boxes)      # [B, 4]
    target_xyxy = xywh2xyxy(target_boxes)  # [B, 4]

    # Tính IoU cho từng cặp (element-wise, không phải pairwise)
    iou = bbox_iou(pred_xyxy, target_xyxy)  # [B]

    # Accuracy@0.5: đếm số sample có IoU ≥ 0.5
    accu = (iou >= 0.5).float().sum().item()

    return iou, accu


def trans_vg_eval_test(pred_boxes, target_boxes):
    """
    Đánh giá toàn bộ test set — dùng trong test.py.
    (Gom tất cả predictions trước, tính 1 lần)

    Args:
        pred_boxes:   [N, 4] — TẤT CẢ predictions (gom từ mọi batch)
        target_boxes: [N, 4] — TẤT CẢ targets

    Returns:
        accu_num: int — tổng số samples có IoU ≥ 0.5
    """
    pred_xyxy = xywh2xyxy(pred_boxes)
    target_xyxy = xywh2xyxy(target_boxes)
    iou = bbox_iou(pred_xyxy, target_xyxy)  # [N]
    accu_num = (iou >= 0.5).float().sum().item()
    return int(accu_num)


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test evaluate.py ===\n")

    # Test loss
    pred = torch.tensor([[0.5, 0.5, 0.3, 0.4],
                          [0.3, 0.3, 0.2, 0.2]])
    target = torch.tensor([[0.5, 0.5, 0.3, 0.4],   # trùng hoàn toàn
                            [0.7, 0.7, 0.2, 0.2]])  # hoàn toàn khác

    losses = trans_vg_loss(pred, target)
    print(f"loss_bbox: {losses['loss_bbox']:.4f}")
    print(f"loss_giou: {losses['loss_giou']:.4f}")

    # Test eval
    iou, accu = trans_vg_eval_val(pred, target)
    print(f"\nIoU per sample: {iou}")
    print(f"Accu (IoU>=0.5): {accu}")

    # Test eval_test
    accu_num = trans_vg_eval_test(pred, target)
    print(f"Total correct:   {accu_num} / {pred.shape[0]}")

    print("\n✅ evaluate.py test passed!")
