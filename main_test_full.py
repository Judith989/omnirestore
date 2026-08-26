# omnirestore_cdd11_test.py
# -*- coding: utf-8 -*-
"""
Full test-set evaluation for WADNet on Derecho.
Fixes: 
 - Data pathing for scratch space
 - PyTorch 2.6 security (weights_only=False)
 - ImageNet un-normalization for correct PSNR
 - Class-wise metric tracking
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torchvision.utils import save_image as imwrite
import numpy as np
from fvcore.nn import FlopCountAnalysis
import lpips

from utils.dataset_loader import load_combined_dataset, IMAGENET_MEAN, IMAGENET_STD
from utils.utils import print_args, tensor_metric, seed_all
from model.embedder import build_embedder
from model.wadt_net import WADNet

# -------------------------
# helpers
# -------------------------
def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict): return state_dict
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}

def _safe_load_component(component, weights, name):
    if weights is None: raise ValueError(f"Missing weights for {name}")
    component.load_state_dict(_strip_module_prefix(weights), strict=False)

def imagenet_unnorm_torch(x: torch.Tensor) -> torch.Tensor:
    """Brings normalized data back to [0,1] for metric calculation."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)

def compute_gflops(model, device, h, w):
    dummy_img = torch.randn(1, 3, h, w).to(device).contiguous()
    dummy_emb = torch.randn(1, 512).to(device).contiguous()
    return FlopCountAnalysis(model, (dummy_img, dummy_emb)).total() / 1e9

# -------------------------
# main eval
# -------------------------
@torch.no_grad()
def evaluate(loader, restorer, embedder, projection, bottleneck, device, args):
    restorer.eval(); projection.eval(); embedder.eval()
    lp_model = lpips.LPIPS(net='alex').to(device).eval()
    
    stats = {} 
    os.makedirs(args.output, exist_ok=True)
    saved = 0
    
    print("🚀 Starting Evaluation Loop...")
    for batch in loader:
        inp = batch.get('input').to(device)
        gt = batch.get('target').to(device)
        styles = batch.get('source_weather', ['clear'] * inp.shape[0])
        files = batch.get('file', [f"img_{i}" for i in range(inp.shape[0])])

        # 1. Conditioning Path
        t_emb = embedder.embed_text_for_style(styles)
        i_emb = embedder.embed_for_style_transfer(inp)
        fused = torch.cat([t_emb, i_emb], dim=-1)
        emb = bottleneck(projection(fused))

        # 2. Restoration
        out, *_ = restorer(inp, emb)

        # 3. Un-normalize for Metrics (fixes the 7.9 PSNR issue)
        out01 = imagenet_unnorm_torch(out)
        gt01 = imagenet_unnorm_torch(gt)
        inp01 = imagenet_unnorm_torch(inp)

        for i in range(inp.shape[0]):
            style = styles[i]
            if style not in stats: stats[style] = {'psnr': [], 'ssim': []}
            
            p = float(tensor_metric(gt01[i:i+1], out01[i:i+1], "PSNR", 1))
            s = float(tensor_metric(gt01[i:i+1], out01[i:i+1], "SSIM", 1))
            
            stats[style]['psnr'].append(p)
            stats[style]['ssim'].append(s)

            if args.save_n > 0 and saved < args.save_n:
                triplet = torch.cat([inp01[i:i+1], out01[i:i+1], gt01[i:i+1]], dim=3)
                base = os.path.splitext(os.path.basename(str(files[i])))[0]
                imwrite(triplet.clamp(0,1), os.path.join(args.output, f"{base}_IOG.png"))
                saved += 1

    print("\n" + "="*45 + "\nCDD-11 CLASS-WISE RESULTS\n" + "="*45)
    all_psnr, all_ssim = [], []
    for style, m in stats.items():
        avg_p, avg_s = np.mean(m['psnr']), np.mean(m['ssim'])
        all_psnr.append(avg_p); all_ssim.append(avg_s)
        print(f"{style:20s} | PSNR: {avg_p:.4f} | SSIM: {avg_s:.4f}")
    
    print("="*45)
    print(f"OVERALL AVERAGE | PSNR: {np.mean(all_psnr):.4f} | SSIM: {np.mean(all_ssim):.4f}\n")

def main(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Embedder
    print(f"✨ > Loading embedder on {device}...")
    embedder = build_embedder(backbone=args.type_name, out_dim=512).to(device)
    e_ckpt = torch.load(args.embedder_model_path, map_location=device, weights_only=False)
    embedder.load_state_dict(_strip_module_prefix(e_ckpt.get("state_dict", e_ckpt)))
    embedder.eval()

    # Load Restorer Components
    projection = nn.Linear(1024, 512).to(device)
    bottleneck = nn.Identity().to(device)
    restorer = WADNet(channel=args.channel, embed_dim=512, window_size=args.window_size, 
                      use_windowed_sa=True, use_shifted_wsa=args.use_shifted_wsa).to(device)
    
    print("💾 > Loading WADNet checkpoint...")
    ckpt = torch.load(args.restore_ckpt, map_location=device, weights_only=False)
    m_states = ckpt.get("model_states", ckpt)
    _safe_load_component(restorer, m_states.get("restorer"), "restorer")
    _safe_load_component(projection, m_states.get("projection"), "projection")
    restorer.eval()

    print(f"Params: {sum(p.numel() for p in restorer.parameters())/1e6:.3f}M | GFLOPs: {compute_gflops(restorer, device, 224, 224):.3f}")

    # -------- DATASET LOADING --------
    # Using absolute scratch path to avoid AssertionError
    data_root = "/glade/derecho/scratch/njoku/CVPR/data/cdd11"
    print(f"📦 > Loading dataset from: {data_root}")

    _, _, test_loader, _ = load_combined_dataset(
        cdd_train_root=data_root, 
        cdd_test_root=data_root, 
        cdd_val_ratio=args.cdd_val_ratio,
        batch_size=args.bs, 
        workers=args.num_works, 
        normalize=True,
        image_size=(args.image_size_h, args.image_size_w)
    )

    if test_loader:
        evaluate(test_loader, restorer, embedder, projection, bottleneck, device, args)
    else:
        print(f"❌ Still failing. Verify splits/cdd11_test.txt exists in {data_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder-model-path", type=str, required=True)
    parser.add_argument("--restore-ckpt", type=str, required=True)
    parser.add_argument("--cdd-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--num-works", type=int, default=4)
    parser.add_argument("--type_name", type=str, default="resnet18")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--channel", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--use-shifted-wsa", action="store_true")
    parser.add_argument("--image-size-h", type=int, default=224)
    parser.add_argument("--image-size-w", type=int, default=224)
    parser.add_argument("--cdd-val-ratio", type=float, default=0.1)
    parser.add_argument("--save-n", type=int, default=0)
    
    args = parser.parse_args()
    main(args)