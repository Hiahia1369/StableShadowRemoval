import torch
import os
import torchvision.transforms as transforms
import math

class Augment_RGB_torch:
### rotate and flip
    def __init__(self, rotate=0):
        self.rotate = rotate
        pass

    def _calculate_scale_factor(self, angle):
        # 将角度转换为弧度
        rad = math.radians(angle)
        # 计算放大倍数
        scale_factor = 1 / (1 - abs(math.sin(rad)))
        return scale_factor
    
    def transform0(self, torch_tensor):
        return torch_tensor  

    def transform1(self, torch_tensor):
        H, W = torch_tensor.shape[1], torch_tensor.shape[2]
        scale_factor = self._calculate_scale_factor(self.rotate)
        train_transform = transforms.Compose([
        transforms.RandomRotation((self.rotate, self.rotate), interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
        transforms.Resize((int(H * scale_factor), int(W * scale_factor)), antialias=True),
        # CenterCrop，如果 size 大于原始尺寸，多余部分将用黑色 (即像素值为 0) 填充
        transforms.CenterCrop([H, W])
        ])
        return train_transform(torch_tensor)
   
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform8(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor
    