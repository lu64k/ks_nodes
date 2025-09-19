import torch
import os
from sklearn.cluster import MiniBatchKMeans
import numpy as np  # 用于处理 NumPy 数组
import cv2
from PIL import Image, ImageOps
from comfy.utils import ProgressBar, common_upscale

class KS_NaturalSaturationAdjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像，格式为 torch.Tensor
                "intensity": ("FLOAT", {  # 控制自然饱和度的调整强度
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_natural_saturation"

    CATEGORY = "sikai nodes/postprocess"
    def adjust_natural_saturation(self, image: torch.Tensor, intensity: float = 1.0):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            # 转换图像为 numpy 格式，图像值为 [0, 255] 的 8 位整数
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()

            # 转换到 HSV 颜色空间
            hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # 拆分 HSV 通道
            h, s, v = cv2.split(hsv_image)

            # 对饱和度通道 (S) 进行非线性调整
            s = s.astype(np.float32) / 255.0  # 正则化到 [0, 1] 范围

            # 自然饱和度公式：增强低饱和度，平滑高饱和度
            s = np.where(s < 0.5, s * (1 + intensity), s + (1 - s) * intensity * 0.3)

            # 限制饱和度值在 [0, 1] 之间
            s = np.clip(s, 0, 1)

            # 将饱和度恢复到 [0, 255] 范围并转换回 uint8 类型
            s = (s * 255).astype(np.uint8)

            # 合并调整后的 HSV 通道
            hsv_adjusted = cv2.merge([h, s, v])

            # 转换回 RGB 颜色空间
            rgb_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)

            # 将结果图像转换为 torch.Tensor 格式并归一化
            result[b] = torch.tensor(rgb_adjusted).float() / 255.0

        return (result,)

class KS_Load_Images_From_Folder:
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_extension": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING",)
    RETURN_NAMES = ("image", "mask", "count", "image_path", "file_names",)
    FUNCTION = "load_images"

    def load_images(self, folder, image_load_cap, start_index, include_extension):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        dir_files = os.listdir(folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # 过滤有效扩展名
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(folder, x) for x in dir_files]

        # 从 start_index 开始
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_path_list = []

        limit_images = (image_load_cap > 0)
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            try:
                i = Image.open(image_path)
            except Exception as e:
                continue
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_path_list.append(image_path)
            image_count += 1

        # 生成文件名列表，根据 include_extension 参数决定是否保留扩展名
        file_names = []
        for path in image_path_list:
            base = os.path.basename(path)
            if not include_extension:
                base, _ = os.path.splitext(base)
            file_names.append(base)
        # 使用 join 将列表转换为逗号分隔的字符串
        file_names_str = ", ".join(file_names)

        if len(images) == 1:
            return (images[0], masks[0], 1, str(image_path_list), file_names_str)

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks[1:]:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0),
                                                                  size=(image1.shape[2], image1.shape[1]),
                                                                  mode='bilinear',
                                                                  align_corners=False)
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            return (image1, mask1, len(images), str(image_path_list), file_names_str)