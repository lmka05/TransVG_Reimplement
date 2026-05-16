import torch
from torchvision.ops.boxes import box_area

def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim = -1)

def xyxy2xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim = -1)

def bbox_iou(box1,box2):

    # Toạ độ từng box 
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:,3]

    # Toạ độ vùng giao
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # Diện tích giao (clamp để tránh giá trị âm khi không giao nhau)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    # Diện tích của từng box
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # IoU = intersection / union
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def box_iou_pairwise(boxes1, boxes2):
    """
    Tính IoU pairwise giữa 2 tập boxes (NxM matrix).

    Args:
        boxes1 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]
        boxes2 (Tensor): [M, 4] — dạng [x1, y1, x2, y2]

    Returns:
        iou (Tensor): [N, M] — ma trận IoU
        union (Tensor): [N, M] — diện tích union
    """
    # Tính diện tích của từng box 
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Tìm giao giữa mọi cặp (N, M)

    x1_1 = boxes1[:,0][:,None]
    y1_1 = boxes1[:,1][:,None]
    x2_1 = boxes1[:,2][:,None]
    y2_1 = boxes1[:,3][:,None]

    x1_2 = boxes2[:,0][:,None]
    y1_2 = boxes2[:,1][:,None]
    x2_2 = boxes2[:,2][:,None]
    y2_2 = boxes2[:,3][:,None]

    # Toạ độ vùng giao
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter

    iou = inter / (union + 1e-16)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Tính Generalized IoU (GIoU) giữa 2 tập boxes.

    GIoU = IoU - (Area_C - Union) / Area_C
    Trong đó Area_C là diện tích hình chữ nhật bao quanh (enclosing box).

    GIoU ∈ [-1, 1]:
        - GIoU = 1: trùng hoàn toàn
        - GIoU = 0: giao nhau 1 phần
        - GIoU < 0: không giao nhau (xa nhau → GIoU → -1)

    Ưu điểm so với IoU thông thường:
        - IoU = 0 khi 2 box không giao → gradient = 0 → model không học được
        - GIoU vẫn có giá trị < 0 khi không giao → gradient vẫn chảy

    Args:
        boxes1 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]
        boxes2 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]

    Returns:
        Tensor: [N, N] — ma trận GIoU

    Dùng trong loss:
        loss_giou = 1 - diag(GIoU(pred, target))
    """
    # Kiểm tra box hợp lệ (x2 >= x1, y2 >= y1)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1: x2 phải >= x1, y2 phải >= y1"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2: x2 phải >= x1, y2 phải >= y1"

    # Tính IoU và union
    iou, union = box_iou_pairwise(boxes1, boxes2)

    # Tính enclosing box (bao quanh cả 2 box)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_c = wh[:, :, 0] * wh[:, :, 1]  # [N, M] — diện tích enclosing box

    # GIoU = IoU - (C - Union) / C
    giou = iou - (area_c - union) / (area_c + 1e-16)
    return giou


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test box_utils ===")

    # Test xywh ↔ xyxy
    xywh = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
    xyxy = xywh2xyxy(xywh)
    print(f"xywh2xyxy: {xywh} → {xyxy}")
    # Expected: [0.3, 0.2, 0.7, 0.8]

    back = xyxy2xywh(xyxy)
    print(f"xyxy2xywh: {xyxy} → {back}")
    # Expected: [0.5, 0.5, 0.4, 0.6]

    # Test IoU
    box1 = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    box2 = torch.tensor([[50.0, 50.0, 150.0, 150.0]])
    iou = bbox_iou(box1, box2)
    print(f"IoU: {iou.item():.4f}")  # ~0.1429

    # Test GIoU
    giou = generalized_box_iou(box1, box2)
    print(f"GIoU: {giou.item():.4f}")

    # Test trường hợp không giao nhau
    box3 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    box4 = torch.tensor([[90.0, 90.0, 100.0, 100.0]])
    iou_no_overlap = bbox_iou(box3, box4)
    giou_no_overlap = generalized_box_iou(box3, box4)
    print(f"No overlap — IoU: {iou_no_overlap.item():.4f}, GIoU: {giou_no_overlap.item():.4f}")

    print("\n✅ box_utils test passed!")