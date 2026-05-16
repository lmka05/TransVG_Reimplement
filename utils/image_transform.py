# ==============================================================================
# image_transforms.py — Xử lý ảnh cho TransVG
# ==============================================================================
# Tất cả tác vụ liên quan đến biến đổi ảnh + bbox đồng thời:
#
#   TRAINING pipeline:
#     RandomSelect(
#       path1: RandomResize([640, 608, 576, ...])
#       path2: RandomResize([400,500,600]) → RandomSizeCrop → RandomResize([640,...])
#     )
#     → ColorJitter
#     → GaussianBlur (optional)
#     → RandomHorizontalFlip
#     → ToTensor
#     → NormalizeAndPad (về 640×640, normalize ImageNet, tạo mask)
#
#   VAL/TEST pipeline:
#     RandomResize([640])
#     → ToTensor
#     → NormalizeAndPad (về 640×640, normalize ImageNet, tạo mask)
#
# Đặc biệt: Mỗi transform nhận và trả về 1 dict:
#   {'img': PIL.Image, 'box': Tensor[4], 'text': str, 'mask': Tensor}
#   → bbox LUÔN được biến đổi đồng bộ với ảnh
#
# Output cuối:
#   - img:  Tensor [3, 640, 640] — normalized + padded
#   - mask: Tensor [640, 640]    — 0 = pixel thật, 1 = padding
#   - box:  Tensor [4]           — normalized [x_c, y_c, w, h] ∈ [0,1]
# ==============================================================================

import math
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageFilter

from .box_utils import xyxy2xywh


# ==============================================================================
# PHẦN 1: CÁC HÀM RESIZE CƠ BẢN
# ==============================================================================

def resize_by_long_side(img, box, size):
    """
    Resize ảnh sao cho cạnh DÀI NHẤT = size. Giữ tỉ lệ.
    Bbox được scale theo cùng tỉ lệ.

    Args:
        img (PIL.Image): Ảnh gốc
        box (Tensor): [4] — dạng [x1, y1, x2, y2]
        size (int): Kích thước cạnh dài nhất mong muốn

    Returns:
        img (PIL.Image): Ảnh đã resize
        box (Tensor): Bbox đã scale

    Ví dụ:
        Ảnh 800×600, size=640
        ratio = 640/800 = 0.8
        → Ảnh mới: 640×480
        → Bbox scale * 0.8
    """
    h, w = img.height, img.width
    ratio = float(size / float(max(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio
    return img, box


def resize_by_short_side(img, box, size):
    """
    Resize ảnh sao cho cạnh NGẮN NHẤT = size. Giữ tỉ lệ.

    Dùng trong bước crop augmentation — resize nhỏ trước khi crop.

    Args:
        img (PIL.Image): Ảnh gốc
        box (Tensor): [4] — [x1, y1, x2, y2]
        size (int): Kích thước cạnh ngắn nhất

    Returns:
        img, box đã resize
    """
    h, w = img.height, img.width
    ratio = float(size / float(min(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio
    return img, box


def crop(image, box, region):
    """
    Crop ảnh theo region và điều chỉnh bbox tương ứng.

    Args:
        image (PIL.Image): Ảnh gốc
        box (Tensor): [4] — [x1, y1, x2, y2]
        region (tuple): (i, j, h, w) — vị trí và kích thước vùng crop
            i = top, j = left, h = height, w = width

    Returns:
        cropped_image, cropped_box
    """
    cropped_image = F.crop(image, *region)

    i, j, h, w = region

    # Dịch bbox theo offset crop
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_box = box - torch.as_tensor([j, i, j, i])

    # Clip bbox để không vượt quá vùng crop
    cropped_box = torch.min(cropped_box.reshape(2, 2), max_size)
    cropped_box = cropped_box.clamp(min=0)
    cropped_box = cropped_box.reshape(-1)

    return cropped_image, cropped_box


# ==============================================================================
# PHẦN 2: CÁC TRANSFORM CLASS (mỗi class nhận/trả dict)
# ==============================================================================

class Compose:
    """
    Gom nhiều transforms thành 1 pipeline tuần tự.
    Giống torchvision.transforms.Compose nhưng hoạt động trên dict.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict):
        for t in self.transforms:
            input_dict = t(input_dict)
        return input_dict

    def __repr__(self):
        lines = [f"  {t}" for t in self.transforms]
        return f"{self.__class__.__name__}(\n" + "\n".join(lines) + "\n)"


class RandomResize:
    """
    Resize ảnh về 1 trong các kích thước cho trước (random chọn).

    Khi aug_scale=True, sizes = [640, 608, 576, 544, 512, 480, 448]
    → mỗi epoch, mỗi ảnh được resize về kích thước khác nhau.
    """
    def __init__(self, sizes, with_long_side=True):
        self.sizes = sizes
        self.with_long_side = with_long_side

    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']
        size = random.choice(self.sizes)

        if self.with_long_side:
            img, box = resize_by_long_side(img, box, size)
        else:
            img, box = resize_by_short_side(img, box, size)

        input_dict['img'] = img
        input_dict['box'] = box
        return input_dict


class RandomSizeCrop:
    """
    Random crop ảnh, đảm bảo tâm bbox nằm trong vùng crop.

    Cơ chế: Thử random crop tối đa max_try lần.
    Chỉ chấp nhận nếu tâm bbox nằm trong vùng crop.
    Nếu không tìm được → giữ nguyên ảnh.
    """
    def __init__(self, min_size, max_size, max_try=20):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try

    def __call__(self, input_dict):
        import torchvision.transforms as T

        img = input_dict['img']
        box = input_dict['box']

        for _ in range(self.max_try):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])

            # Kiểm tra tâm bbox có nằm trong vùng crop không
            box_xywh = xyxy2xywh(box)
            box_cx, box_cy = box_xywh[0], box_xywh[1]
            if box_cx > region[1] and box_cy > region[0]:
                img, box = crop(img, box, region)
                input_dict['img'] = img
                input_dict['box'] = box
                return input_dict

        # Không tìm được vùng crop hợp lệ → giữ nguyên
        return input_dict


class RandomSelect:
    """
    Random chọn giữa 2 transform pipelines.

    Đặc biệt: Nếu câu text chứa từ chỉ hướng (left, right, top, bottom, middle)
    → LUÔN chọn transforms1 (resize thường, không crop)
    → Vì crop có thể làm mất context về vị trí.
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, input_dict):
        text = input_dict['text']

        # Nếu câu chứa từ chỉ hướng → không crop
        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        for word in dir_words:
            if word in text:
                return self.transforms1(input_dict)

        # Random chọn
        if random.random() < self.p:
            return self.transforms2(input_dict)
        else:
            return self.transforms1(input_dict)


class ColorJitter:
    """
    Random thay đổi brightness, contrast, saturation.
    Áp dụng với xác suất 80%.
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, input_dict):
        if random.random() < 0.8:
            img = input_dict['img']

            # Random thứ tự áp dụng
            for func_id in np.random.permutation(3):
                if func_id == 0:
                    factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                    img = ImageEnhance.Brightness(img).enhance(factor)
                elif func_id == 1:
                    factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                    img = ImageEnhance.Contrast(img).enhance(factor)
                elif func_id == 2:
                    factor = random.uniform(1 - self.saturation, 1 + self.saturation)
                    img = ImageEnhance.Color(img).enhance(factor)

            input_dict['img'] = img

        return input_dict


class GaussianBlur:
    """
    Áp dụng Gaussian blur ngẫu nhiên (với xác suất p).
    Thường tắt (aug_blur=False).
    """
    def __init__(self, sigma=(0.1, 2.0), aug_blur=False):
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.0

    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict['img']
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            input_dict['img'] = img
        return input_dict


class RandomHorizontalFlip:
    """
    Lật ngang ảnh với xác suất 50%.

    Quan trọng: Khi lật ảnh, phải:
      1. Lật bbox tương ứng
      2. Swap "left" ↔ "right" trong text
         (vì sau khi lật, "trái" thành "phải" và ngược lại)
    """
    def __call__(self, input_dict):
        if random.random() < 0.5:
            img = input_dict['img']
            box = input_dict['box']
            text = input_dict['text']

            # Lật ảnh
            img = F.hflip(img)

            # Swap left ↔ right trong text
            text = text.replace('right', '*TEMP*').replace('left', 'right').replace('*TEMP*', 'left')

            # Lật bbox: x1_new = w - x2_old, x2_new = w - x1_old
            h, w = img.height, img.width
            box = box[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + \
                  torch.as_tensor([w, 0, w, 0])

            input_dict['img'] = img
            input_dict['box'] = box
            input_dict['text'] = text

        return input_dict


class ToTensor:
    """
    Chuyển PIL Image → Tensor [C, H, W] (float32, [0, 1]).
    """
    def __call__(self, input_dict):
        img = input_dict['img']
        img = F.to_tensor(img)  # PIL → Tensor [C, H, W], range [0, 1]
        input_dict['img'] = img
        return input_dict


class NormalizeAndPad:
    """
    Normalize ảnh bằng ImageNet stats + Pad về kích thước cố định.

    Bước này rất QUAN TRỌNG vì:
      1. Normalize: ResNet pretrained trên ImageNet cần input đã normalize
      2. Pad: Tất cả ảnh trong batch phải cùng kích thước
      3. Mask: Đánh dấu vùng padding (1) vs vùng ảnh thật (0)
         → BERT/Transformer sẽ ignore vùng padding
      4. Bbox: Chuyển sang normalized [x_c, y_c, w, h] ∈ [0, 1]

    Output:
        img:  [3, size, size] — normalized + padded
        mask: [size, size]    — 0 = ảnh thật, 1 = padding
        box:  [4]             — normalized [x_c, y_c, w, h]
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 size=640, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate

    def __call__(self, input_dict):
        img = input_dict['img']  # Tensor [C, H, W]

        # 1. Normalize bằng ImageNet mean/std
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]
        dw = self.size - w  # Lượng pad ngang
        dh = self.size - h  # Lượng pad dọc

        # 2. Tính vị trí đặt ảnh trong canvas
        if self.aug_translate:
            # Training: random dịch vị trí ảnh (data augmentation)
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            # Val/Test: đặt ảnh ở giữa
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        # 3. Tạo canvas + mask
        out_img = torch.zeros((3, self.size, self.size), dtype=torch.float32)
        out_mask = torch.ones((self.size, self.size), dtype=torch.int32)

        out_img[:, top:top+h, left:left+w] = img
        out_mask[top:top+h, left:left+w] = 0  # 0 = ảnh thật

        input_dict['img'] = out_img
        input_dict['mask'] = out_mask

        # 4. Điều chỉnh bbox
        if 'box' in input_dict:
            box = input_dict['box']
            # Dịch bbox theo offset
            box[0] += left
            box[2] += left
            box[1] += top
            box[3] += top

            # Chuyển sang [x_c, y_c, w, h] và normalize về [0, 1]
            out_h, out_w = out_img.shape[-2:]
            box = xyxy2xywh(box)
            box = box / torch.tensor([out_w, out_h, out_w, out_h], dtype=torch.float32)
            input_dict['box'] = box

        return input_dict


# ==============================================================================
# PHẦN 3: HÀM FACTORY — TẠO TRANSFORM PIPELINE
# ==============================================================================

def make_transforms(config, split):
    """
    Tạo transform pipeline phù hợp cho từng split.

    Args:
        config: Config object
        split (str): 'train', 'val', 'test', 'testA', 'testB'

    Returns:
        Compose: Pipeline transforms

    Pipeline:
        train: MultiScale + Crop + ColorJitter + Blur + Flip + Normalize + Pad
        val/test: Resize(640) + Normalize + Pad
    """
    imsize = config.imsize

    if split == 'train':
        # Multi-scale sizes: [640, 608, 576, 544, 512, 480, 448]
        scales = []
        if config.aug_scale:
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        crop_prob = 0.5 if config.aug_crop else 0.0

        return Compose([
            RandomSelect(
                # Path 1: Resize theo long side
                RandomResize(scales),
                # Path 2: Resize nhỏ → Crop → Resize lại
                Compose([
                    RandomResize([400, 500, 600], with_long_side=False),
                    RandomSizeCrop(384, 600),
                    RandomResize(scales),
                ]),
                p=crop_prob
            ),
            ColorJitter(0.4, 0.4, 0.4),
            GaussianBlur(aug_blur=config.aug_blur),
            RandomHorizontalFlip(),
            ToTensor(),
            NormalizeAndPad(size=imsize, aug_translate=config.aug_translate),
        ])

    if split in ['val', 'test', 'testA', 'testB']:
        return Compose([
            RandomResize([imsize]),
            ToTensor(),
            NormalizeAndPad(size=imsize),
        ])

    raise ValueError(f"Unknown split: {split}")


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    from PIL import Image
    print("=== Test ImageTransforms ===\n")

    # Tạo ảnh giả 800×600 (RGB)
    fake_img = Image.fromarray(np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8))
    fake_box = torch.tensor([100.0, 50.0, 400.0, 300.0])  # [x1, y1, x2, y2]

    input_dict = {
        'img': fake_img,
        'box': fake_box.clone(),
        'text': "the man on the left"
    }

    # Test val pipeline (đơn giản hơn)
    class FakeConfig:
        imsize = 640
        aug_scale = True
        aug_crop = True
        aug_translate = True
        aug_blur = False

    val_transform = make_transforms(FakeConfig, 'val')
    result = val_transform({
        'img': fake_img.copy(),
        'box': fake_box.clone(),
        'text': "the man on the left"
    })

    print(f"Val output img shape:  {result['img'].shape}")   # [3, 640, 640]
    print(f"Val output mask shape: {result['mask'].shape}")   # [640, 640]
    print(f"Val output box:        {result['box']}")          # normalized xywh
    print(f"Val mask unique:       {result['mask'].unique()}") # [0, 1]

    # Test train pipeline
    train_transform = make_transforms(FakeConfig, 'train')
    result_train = train_transform({
        'img': fake_img.copy(),
        'box': fake_box.clone(),
        'text': "the man on the left"
    })
    print(f"\nTrain output img shape:  {result_train['img'].shape}")
    print(f"Train output box:        {result_train['box']}")

    print("\n✅ ImageTransforms test passed!")
