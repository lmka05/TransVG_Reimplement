# ==============================================================================
# misc.py — Tiện ích chung cho TransVG
# ==============================================================================
# Các helper không thuộc về image/text/box:
#   - NestedTensor: Wrapper cho tensor + mask (dùng cho cả image và text)
#   - collate_fn: Custom collate cho DataLoader
#   - SmoothedValue & MetricLogger: Theo dõi loss/accuracy khi training
# ==============================================================================

import time
import datetime
from collections import defaultdict, deque
from typing import Optional, List

import torch
from torch import Tensor


# ==============================================================================
# PHẦN 1: NESTED TENSOR
# ==============================================================================

class NestedTensor:
    """
    Wrapper gom tensor + mask thành 1 object.

    Dùng cho cả image và text:
      - Image: tensors = [B, 3, H, W], mask = [B, H, W]
               mask[i, y, x] = 0 → pixel thật, 1 → padding
      - Text:  tensors = [B, max_len], mask = [B, max_len]
               mask[i, j] = 1 → token thật, 0 → padding
               (Chú ý: BERT mask ngược với image mask!)

    Lý do cần class này:
      - Transformer cần biết vùng nào là padding để ignore trong attention
      - Gom tensor + mask vào 1 object → code gọn hơn
    """

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """Di chuyển cả tensor và mask sang device (CPU/GPU)."""
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        """Tách tensor và mask ra riêng."""
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# ==============================================================================
# PHẦN 2: COLLATE FUNCTION
# ==============================================================================

def collate_fn(raw_batch):
    """
    Custom collate function cho DataLoader.

    Dataset trả về mỗi sample:
        (img, img_mask, word_id, word_mask, bbox)

    Collate gom thành batch:
        img_data: NestedTensor(img=[B,3,640,640], mask=[B,640,640])
        text_data: NestedTensor(word_id=[B,17], word_mask=[B,17])
        bbox: Tensor [B, 4]

    Args:
        raw_batch: list of tuples từ Dataset.__getitem__

    Returns:
        tuple: (img_data, text_data, bbox)
    """
    raw_batch = list(zip(*raw_batch))

    # Image: stack thành batch tensor + mask
    img = torch.stack(raw_batch[0])           # [B, 3, 640, 640]
    img_mask = torch.stack(raw_batch[1])      # [B, 640, 640]
    img_data = NestedTensor(img, img_mask)

    # Text: stack word_ids + attention_mask
    word_id = torch.stack(raw_batch[2])       # [B, 17]
    word_mask = torch.stack(raw_batch[3])     # [B, 17]
    text_data = NestedTensor(word_id, word_mask)

    # Bbox
    bbox = torch.stack(raw_batch[4])          # [B, 4]

    return img_data, text_data, bbox


# ==============================================================================
# PHẦN 3: LOGGING — Theo dõi metrics khi training
# ==============================================================================

class SmoothedValue:
    """
    Theo dõi 1 giá trị (loss, accuracy) qua thời gian.

    Cung cấp các thống kê:
      - median: Giá trị trung vị (ổn định, ít bị ảnh hưởng outlier)
      - avg:    Trung bình trong window gần nhất
      - global_avg: Trung bình toàn bộ lịch sử
      - max:    Giá trị lớn nhất trong window
      - value:  Giá trị mới nhất

    window_size: Số giá trị gần nhất để tính avg/median
    """

    def __init__(self, window_size=50, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 1e-12  # Tránh chia cho 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Thêm giá trị mới."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger:
    """
    Quản lý nhiều SmoothedValue cùng lúc (loss, accuracy, lr, ...).

    Cách dùng:
        logger = MetricLogger()
        logger.update(loss=0.5, accuracy=0.8)
        logger.update(loss=0.3, accuracy=0.85)
        print(logger)  # → "loss: 0.4000 (0.4000)  accuracy: 0.8250 (0.8250)"
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Cập nhật các metrics."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Expected float/int, got {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def __str__(self):
        parts = [f"{name}: {str(meter)}" for name, meter in self.meters.items()]
        return self.delimiter.join(parts)

    def add_meter(self, name, meter):
        """Thêm 1 SmoothedValue mới."""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Iterator wrapper — in log mỗi print_freq batches.

        Cách dùng:
            for batch in logger.log_every(dataloader, print_freq=50, header="Epoch [1]"):
                # ... train ...
                logger.update(loss=loss_val)

        Output (mỗi 50 batches):
            Epoch [1] [50/1000] eta: 00:05:30  loss: 0.3000 (0.4500)  time: 0.3200  data: 0.0100  max mem: 4500
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / max(len(iterable), 1):.4f} s / it)')


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test misc ===\n")

    # Test NestedTensor
    print("--- NestedTensor ---")
    img = torch.randn(2, 3, 640, 640)
    mask = torch.zeros(2, 640, 640, dtype=torch.int32)
    nt = NestedTensor(img, mask)
    print(f"NestedTensor img shape: {nt.tensors.shape}")
    print(f"NestedTensor mask shape: {nt.mask.shape}")

    t, m = nt.decompose()
    print(f"Decompose: tensor {t.shape}, mask {m.shape}")

    # Test SmoothedValue
    print("\n--- SmoothedValue ---")
    sv = SmoothedValue(window_size=5)
    for v in [0.5, 0.4, 0.3, 0.35, 0.32]:
        sv.update(v)
    print(f"SmoothedValue: {sv}")
    print(f"  median={sv.median:.4f}, avg={sv.avg:.4f}, global_avg={sv.global_avg:.4f}")

    # Test MetricLogger
    print("\n--- MetricLogger ---")
    logger = MetricLogger()
    logger.update(loss=0.5, acc=0.7)
    logger.update(loss=0.3, acc=0.8)
    print(f"Logger: {logger}")

    print("\n✅ misc test passed!")
