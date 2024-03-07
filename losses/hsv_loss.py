import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from functools import partial
import kornia
import utils.funcs as f

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class HSV_Difference(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img1, img2):

        # Convert images to PyTorch tensors
        img1 = torch.from_numpy(img1 / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img2 = torch.from_numpy(img2 / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Denormalize images
        #img1 = f.denormalization_torch(img1, device)
        #img2 = f.denormalization_torch(img2, device)

        # Convert images to HSV
        hsv_img1 = kornia.color.rgb_to_hsv(img1)
        hsv_img2 = kornia.color.rgb_to_hsv(img2)

        # Compute absolute differences in HSV channels
        h_diff = torch.abs(hsv_img1[:, 0, :, :] - hsv_img2[:, 0, :, :])
        s_diff = torch.abs(hsv_img1[:, 1, :, :] - hsv_img2[:, 1, :, :])
        v_diff = torch.abs(hsv_img1[:, 2, :, :] - hsv_img2[:, 2, :, :])

        # Normalize differences
        h_diff = h_diff / (2 * 3.1416)  # Normalizing hue difference
        s_diff = s_diff / 1.0  # Normalizing saturation difference
        v_diff = v_diff / 1.0  # Normalizing value difference

        h_diff = h_diff.squeeze().cpu().numpy()
        s_diff = s_diff.squeeze().cpu().numpy()
        v_diff = v_diff.squeeze().cpu().numpy()

        h_diff = gaussian_filter(h_diff, sigma=2)
        s_diff = gaussian_filter(s_diff, sigma=2)
        v_diff = gaussian_filter(v_diff, sigma=2)

        return [h_diff, s_diff, v_diff]


class HSV_Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsv_diff = HSV_Difference()

    def forward(self, img1, img2):
        return self.hsv_diff(img1, img2)