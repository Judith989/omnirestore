# -*- coding: utf-8 -*-
# test_only_images.py
import os, time, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import save_image as imwrite

from utils.dataset_loader import load_combined_dataset, IMAGENET_MEAN, IMAGENET_STD
from model.wadt_net import WADNet
from model.embedder import build_embedder

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips


def print_args(args):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


def strip_module(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def imagenet_unnorm_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W] in ImageNet-normalized space
    returns: [B,3,H,W] in [0,1]
    """
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    y = x * std + mean
    return y.clamp(0, 1)


def lpips_prepare(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: [B,3,H,W] in [0,1]
    returns: [-1,1] for lpips
    """
    return x01.clamp(0, 1) * 2 - 1


def safe_torch_load(path: str, device):
    """
    Handles PyTorch 2.6+ default weights_only=True behavior.
    You trust your own checkpoint, so weights_only=False is appropriate.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # older torch versions
        return torch.load(path, map_location=device)


def load_restorer_from_best(device, best_ckpt_path, channel=16, embed_dim=512, window_size=8, use_windowed_sa=True):
    if not os.path.isfile(best_ckpt_path):
        raise FileNotFoundError(f"best.ckpt not found: {best_ckpt_path}")

    ckpt = safe_torch_load(best_ckpt_path, device)
    model_states = ckpt.get("model_states", ckpt)
    rest_w = model_states.get("restorer", None)
    if rest_w is None:
        raise ValueError("Could not find model_states['restorer'] in best.ckpt")

    restorer = WADNet(channel=channel, embed_dim=embed_dim, window_size=window_size, use_windowed_sa=use_windowed_sa)
    restorer.load_state_dict(strip_module(rest_w), strict=True)
    restorer = restorer.to(device).eval()
    for p in restorer.parameters():
        p.requires_grad = False
    return restorer


@torch.no_grad()
def evaluate_loader(args, loader, restorer, embedder, lpips_fn, device):
    os.makedirs(args.output, exist_ok=True)

    psnr_in_all, ssim_in_all, lp_in_all = [], [], []
    psnr_out_all, ssim_out_all, lp_out_all = [], [], []
    time_record = []

    saved = 0

    for batch in loader:
        inp = batch["input"].to(device)    # ImageNet normalized
        tgt = batch["target"].to(device)   # ImageNet normalized
        files = batch.get("file", [""] * inp.shape[0])

        # --- Image-only conditioning 
        t0 = time.time()
        img_emb = embedder.embed_for_style_transfer(inp)  # <- IMPORTANT: uses dataset tensor directly
        out, *_ = restorer(inp, img_emb)
        dt = time.time() - t0
        time_record.append(dt)

        # Convert to pixel space [0,1] for metrics and saving
        inp01 = imagenet_unnorm_torch(inp)
        out01 = imagenet_unnorm_torch(out)
        tgt01 = imagenet_unnorm_torch(tgt)

        # Metrics per-image (skimage expects HWC numpy)
        for b in range(inp.shape[0]):
            in_np = inp01[b].permute(1, 2, 0).detach().cpu().numpy()
            out_np = out01[b].permute(1, 2, 0).detach().cpu().numpy()
            gt_np = tgt01[b].permute(1, 2, 0).detach().cpu().numpy()

            psnr_in = compute_psnr(gt_np, in_np, data_range=1.0)
            ssim_in = compute_ssim(gt_np, in_np, channel_axis=-1, data_range=1.0)
            lp_in = lpips_fn(lpips_prepare(inp01[b:b+1]), lpips_prepare(tgt01[b:b+1])).item()

            psnr_out = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim_out = compute_ssim(gt_np, out_np, channel_axis=-1, data_range=1.0)
            lp_out = lpips_fn(lpips_prepare(out01[b:b+1]), lpips_prepare(tgt01[b:b+1])).item()

            psnr_in_all.append(psnr_in); ssim_in_all.append(ssim_in); lp_in_all.append(lp_in)
            psnr_out_all.append(psnr_out); ssim_out_all.append(ssim_out); lp_out_all.append(lp_out)

            if (args.save_n is not None) and (saved < args.save_n):
                fn = os.path.basename(str(files[b]))
                if not fn:
                    fn = f"sample_{len(psnr_out_all):06d}.png"
                vis = torch.cat([inp01[b:b+1], out01[b:b+1], tgt01[b:b+1]], dim=3)  # [1,3,H,3W]
                imwrite(vis, os.path.join(args.output, fn))
                saved += 1

    # -------------------------
    # Report (finite-PSNR mean)
    # -------------------------
    def mean_finite(x):
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            return float("nan"), 0, len(x)
        return float(x[finite].mean()), int((~finite).sum()), len(x)

    psnr_in_mean, psnr_in_inf, psnr_in_total = mean_finite(psnr_in_all)
    psnr_out_mean, psnr_out_inf, psnr_out_total = mean_finite(psnr_out_all)

    print("\n====== AVERAGES (FULL TEST SET) ======")
    print(
        f"INPUT  : PSNR={psnr_in_mean:.4f} (inf={psnr_in_inf}/{psnr_in_total}), "
        f"SSIM={float(np.mean(ssim_in_all)):.4f}, LPIPS={float(np.mean(lp_in_all)):.4f}"
    )
    print(
        f"OUTPUT : PSNR={psnr_out_mean:.4f} (inf={psnr_out_inf}/{psnr_out_total}), "
        f"SSIM={float(np.mean(ssim_out_all)):.4f}, LPIPS={float(np.mean(lp_out_all)):.4f}"
    )
    print("IMPROV : ΔPSNR={:.4f}, ΔSSIM={:.4f}, ΔLPIPS={:.4f}".format(
        psnr_out_mean - psnr_in_mean,
        float(np.mean(ssim_out_all) - np.mean(ssim_in_all)),
        float(np.mean(lp_in_all) - np.mean(lp_out_all)),
    ))
    print(f"Avg runtime per batch: {float(np.mean(time_record)):.4f}s over {len(time_record)} batches")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✨ > Loading embedder (frozen)...")
    embedder = build_embedder(backbone=args.type_name, out_dim=512).to(device)
    # Your embedder ckpt loading utility:
    from utils.ckpt_utils import load_checkpoint_if_any
    load_checkpoint_if_any(embedder, ckpt_path=args.embedder_model_path, map_location=device)
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad = False

    print(" > Loading best checkpoint (restorer only)...")
    restorer = load_restorer_from_best(
        device=device,
        best_ckpt_path=args.best_ckpt,
        channel=args.channel,
        embed_dim=512,
        window_size=args.window_size,
        use_windowed_sa=(not args.disable_wsa),
    )

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    # ---- test loader via your dataset loader ----
    _, _, test_loader, eval_info = load_combined_dataset(
        cdd_train_root=args.cdd_root if args.cdd_root != "DUMMY_PATH" else None,
        cdd_test_root=args.cdd_root if args.cdd_root != "DUMMY_PATH" else None,  # <--- common case: same root
        cdd_val_ratio=args.cdd_val_ratio,
        split="test",
        image_size=(args.image_size_h, args.image_size_w),
        batch_size=args.bs,
        workers=args.num_works,
        distributed=False,
        normalize=True,
    )

    if test_loader is None:
        raise RuntimeError("test_loader is None. Check dataset root + test folder structure.")

    print("🚀 > Evaluating full test set (image-only conditioning)...")
    evaluate_loader(args, test_loader, restorer, embedder, lpips_fn, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WADNet full test (image-only conditioning)")
    parser.add_argument("--embedder-model-path", type=str, required=True)
    parser.add_argument("--best-ckpt", type=str, required=True)

    parser.add_argument("--cdd-root", type=str, required=True)
    parser.add_argument("--cdd-val-ratio", type=float, default=0.1)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--num-works", type=int, default=4)
    parser.add_argument("--type_name", type=str, default="resnet18")

    parser.add_argument("--image-size-h", type=int, default=224)
    parser.add_argument("--image-size-w", type=int, default=224)

    parser.add_argument("--channel", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--disable-wsa", action="store_true")

    parser.add_argument("--save-n", type=int, default=40, help="save first N visualizations (in/out/gt concat)")

    args = parser.parse_args()
    print_args(args)
    main(args)