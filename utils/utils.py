# utils/utils.py
# -----------------------------------------------------------------------------
# General utilities for metrics, reproducibility, and evaluation.
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
import math
import random
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim


# -----------------------------------------------------------------------------
# Reproducibility: seed setting
# -----------------------------------------------------------------------------
def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# PSNR & SSIM Computation
# -----------------------------------------------------------------------------
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return 100.0
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    C1, C2 = 0.01**2, 0.03**2
    device = pred.device
    channel = pred.size(1)

    def _gaussian(win_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - win_size//2) ** 2 / (2 * sigma**2))
                             for x in range(win_size)])
        return gauss / gauss.sum()

    window = _gaussian(window_size, 1.5).unsqueeze(1)
    window = window @ window.t()
    window = window.expand(channel, 1, window_size, window_size).to(device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

# -----------------------------------------------------------------------------
# Tensor-based Metric using skimage for batch evaluation with explicit SSIM patch
# -----------------------------------------------------------------------------
def tensor_metric(img: torch.Tensor, imclean: torch.Tensor, model: str, data_range: float = 1.0) -> float:
    img_cpu = img.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)  # B H W C
    imgclean = imclean.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
    SUM = 0
    for i in range(img_cpu.shape[0]):
        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i], img_cpu[i], data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i], img_cpu[i])
        elif model == 'SSIM':
            # Explicit small win_size and channel_axis for robustness
            SUM += compare_ssim(
                imgclean[i],
                img_cpu[i],
                data_range=data_range,
                channel_axis=-1,  # channel last for HWC format
                win_size=3       # smaller window size to handle small images
            )
        else:
            print('tensor_metric: Model False! Model should be PSNR, MSE or SSIM')
    return SUM / img_cpu.shape[0]

# -----------------------------------------------------------------------------
# Excel saving metric results
# -----------------------------------------------------------------------------
def load_excel(x):
    df = pd.DataFrame(x)
    with pd.ExcelWriter('./metric_result.xlsx') as writer:
        df.to_excel(writer, sheet_name='PSNR-SSIM', float_format='%.5f')


# -----------------------------------------------------------------------------
# Learning rate adjustment schedule
# -----------------------------------------------------------------------------
def adjust_learning_rate(optimizer, epoch: int, lr_update_freq: int):
    if epoch != 0 and epoch % lr_update_freq == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2
    return optimizer


# -----------------------------------------------------------------------------
# Data processing helper
# -----------------------------------------------------------------------------
def data_process(data, args, device):
    combine_type = args.degr_type
    b, n, c, w, h = data.size()

    pos_data = data[:, 0, :, :, :]

    inp_data = torch.zeros((b, c, w, h))
    inp_class = []

    neg_data = torch.zeros((b, n - 2, c, w, h))

    index = np.random.randint(1, n, (b,))
    for i in range(b):
        k = 0
        for j in range(n):
            if j == 0:
                continue
            elif index[i] == j:
                inp_class.append(combine_type[index[i]])
                inp_data[i, :, :, :] = data[i, index[i], :, :, :]
            else:
                neg_data[i, k, :, :, :] = data[i, j, :, :, :]
                k += 1

    return pos_data.to(device), [inp_data.to(device), inp_class], neg_data.to(device)


# -----------------------------------------------------------------------------
# Arguments printer
# -----------------------------------------------------------------------------
def print_args(args):
    print("\nArguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("")

@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_extra: bool = False,
    device: torch.device | None = None,
) -> Dict[str, float]:
    device = device or (pred.device if torch.is_tensor(pred) else "cpu")
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    results: Dict[str, float] = {}
    results["psnr"] = compute_psnr(pred, target)
    results["ssim"] = compute_ssim(pred, target)

    if compute_extra:
        try:
            from utils import compute_lpips, compute_dists, compute_niqe  # import from utils or external as needed
            results["lpips"] = compute_lpips(pred, target, device=device)
            results["dists"] = compute_dists(pred, target, device=device)
            results["niqe_pred"] = compute_niqe(pred, device=device)
            results["niqe_gt"] = compute_niqe(target, device=device)
        except ImportError:
            pass

    return results
