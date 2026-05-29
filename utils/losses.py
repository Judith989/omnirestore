###utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torchvision.models import vgg16, VGG16_Weights
import torchvision

# --- Helper functions for MS-SSIM (Device Fixes Applied) ---
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 1 if torch.max(img1) <= 1 else 255
        min_val = 0 if torch.min(img1) >= 0 else -1
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        # DEVICE FIX 1: Ensure window is on the same device as img1
        window = create_window(real_size, channel=channel).to(img1.device) 

    # ... (rest of ssim logic)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    # 🛑 DEVICE FIX 2: Ensure weights are on the same device as img1
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device) 
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, 2)
        img2 = F.avg_pool2d(img2, 2)

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

# --- Auxiliary Loss Classes ---
class ContrastLoss(nn.Module):
    def __init__(self, device=None): # <-- Added device arg
        super().__init__()
        self.l1 = nn.L1Loss()
        device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        #DEVICE FIX 3: VGG moved to device
        self.model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device) 
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {'3': "relu1_2", '8': "relu2_2", '15': "relu3_3"}

    # ... (trim_channels, gen_features, and forward remain the same)
    def trim_channels(self, x):
        if x.shape[1] == 4:
            return x[:, :3, :, :]
        return x

    def gen_features(self, x):
        x = self.trim_channels(x)
        outputs = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outputs.append(x)
        return outputs

    def forward(self, inp, pos, neg, out):
        inp = self.trim_channels(inp)
        pos = self.trim_channels(pos)
        out = self.trim_channels(out)

        neg_feats_list = []
        if len(neg.shape) == 5:
            for i in range(neg.shape[1]):
                neg_slice = neg[:, i, :, :, :]
                neg_slice = self.trim_channels(neg_slice)
                neg_feats_list.append(self.gen_features(neg_slice))
        elif len(neg.shape) == 4:
            neg = self.trim_channels(neg)
            neg_feats_list.append(self.gen_features(neg))
        else:
            raise ValueError(f"Unexpected neg tensor shape: {neg.shape}")

        inp_feats = self.gen_features(inp)
        pos_feats = self.gen_features(pos)
        out_feats = self.gen_features(out)

        loss = 0
        for i in range(len(pos_feats)):
            pos_term = self.l1(out_feats[i], pos_feats[i].detach())
            inp_term = self.l1(out_feats[i], inp_feats[i].detach()) / (len(neg_feats_list) + 1)
            neg_term = sum(self.l1(out_feats[i], nf[i].detach()) for nf in neg_feats_list) / (len(neg_feats_list) + 1)
            loss += pos_term / (inp_term + neg_term + 1e-7)
        return loss / len(pos_feats)

class VGGPerceptualLoss(nn.Module):
    # Standard VGG Perceptual Loss (L1 distance in feature space)
    def __init__(self, layers=[3, 8, 15], weight=1.0, device=None): 
        super().__init__()
        self.layers = set(str(i) for i in layers)
        device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # DEVICE FIX 4: VGG moved to device
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(device) 
        self.weight = weight
        for p in self.vgg.parameters():
            p.requires_grad = False
        
        # Normalization: VGG requires input to be normalized by ImageNet mean/std
        # Define normalization layer on the specified device
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)


    def _feat(self, x):
        # Apply ImageNet normalization before VGG
        h = self.normalize(x)
        feats = []
        for name, module in self.vgg._modules.items():
            h = module(h)
            if name in self.layers:
                feats.append(h)
        return feats

    def forward(self, pred, target):
        if pred is None or target is None:
            return 0.0
        
        # NOTE: Pred and Target should be 3-channel images (which your trim_channels handles implicitly)
        feats_pred = self._feat(pred)
        feats_tgt = self._feat(target)
        # Use L1 loss for feature distance
        loss = sum(F.l1_loss(a, b.detach()) for a, b in zip(feats_pred, feats_tgt)) / len(feats_pred)
        return self.weight * loss

def cosine_style_loss(pred_style, pos_style=None, neg_style=None, w_pos=1.0, w_neg=0.25):
    # ... (cosine_style_loss remains the same)
    if pred_style is None:
        return 0.0
    loss = pred_style.new_zeros(())
    n = 0
    if pos_style is not None:
        sim_pos = F.cosine_similarity(pred_style, pos_style, dim=-1)
        loss = loss + w_pos * (1.0 - sim_pos).mean()
        n += 1
    if neg_style is not None:
        sim_neg = F.cosine_similarity(pred_style, neg_style, dim=-1)
        loss = loss + w_neg * (1.0 + sim_neg).mean()
        n += 1
    if n == 0:
        return pred_style.new_zeros(())
    return loss

def severity_mse(pred, target):
    if pred is None or target is None:
        return 0.0
    return F.mse_loss(pred, target)

# -----------------------------------------------------
# ---------- Deep Supervision Feature Extractor (DEVICE FIX) ----------
class GTFeatureExtractor(nn.Module):
    # Pass device during initialization
    def __init__(self, channels=[16, 32, 64], device=None): 
        super().__init__()
        

        device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.channels = channels


        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1).to(device)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0).to(device)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=1, stride=1, padding=0).to(device)

    def forward(self, x):
        feat_l = self.conv1(x) 
        x = self.maxpool(feat_l) 
        feat_m = self.conv2(x) 
        x = self.maxpool(feat_m)
        feat_s = self.conv3(x) 
        
        # Detaching GT features is good practice for deep supervision targets
        return feat_l.detach(), feat_m.detach(), feat_s.detach()


# -----------------------------------------------------
# ---------- Unified Total Loss Class ----------
class Total_loss(nn.Module):

    def __init__(self, args, device='cuda'): 
        super(Total_loss, self).__init__()
        
        # 1. Unpack 4 weights: L1, MSSSIM, Contrast/DRL, Perceptual
        try:
            # We assume the main script was updated to provide 4 weights: 
            # [w_sl1, w_msssim, w_drl (Contrast), w_percep (VGG)]
            self.w_sl1, self.w_msssim, self.w_drl, self.w_percep = args.loss_weight
        except ValueError:
            print("WARNING: Expected 4 loss_weight arguments but received 3. Using default for w_percep (0.5).")
            # Fallback for old 3-weight configuration, using w_drl as Contrast weight
            self.w_sl1, self.w_msssim, self.w_drl = args.loss_weight
            self.w_percep = 0.5 
            
        self.to(device)
        
        # 2. Initialize Loss Components (passing device)
        self.contrast = ContrastLoss(device=device) # Uses w_drl
        # Use a weight of 1.0 here, the final weight is applied in forward
        self.perceptual = VGGPerceptualLoss(weight=1.0, device=device) 
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

        # 3. Component to generate ground truth features from the target image (pos)
        self.gt_feat_extractor = GTFeatureExtractor(channels=[16, 32, 64], device=device)
        
        # 4. Dedicate a weight for the explicit Deep Supervision Feature Loss (feat_l, m, s)
        # This is a new, separate term, and we'll reuse w_drl as the weight for this too.
        self.w_feat_match = self.w_drl # Reusing the DRL weight for both Contrast and Feature Match for simplicity



    def forward(self, inp, pos, neg, out, feat_l=None, feat_m=None, feat_s=None):
        
        # 1. Existing Image-level Losses
        smooth_loss_l1 = self.smooth_l1_loss(out, pos)
        msssim_loss = 1 - msssim(out, pos, normalize=True)
        # Use Contrast Loss (w_drl)
        c_loss = self.contrast(inp, pos, neg, out) 
        
        # 2. VGG Perceptual Loss (NEWLY DEDICATED TERM)
        vgg_percep_loss = 0.0
        if self.w_percep > 0:
             # Calculate L1 distance in VGG feature space
             vgg_percep_loss = self.perceptual(out, pos)
            
        # 3. Deep Supervision Feature Matching Loss (feat_l, m, s)
        feature_match_loss = 0.0
        
        if feat_l is not None and feat_m is not None and feat_s is not None:
            # Generate Ground Truth Feature Targets
            gt_feat_l, gt_feat_m, gt_feat_s = self.gt_feat_extractor(pos)
            
            # Use L1 Loss for deep supervision
            feature_match_loss += self.l1_loss(feat_l, gt_feat_l)
            feature_match_loss += self.l1_loss(feat_m, gt_feat_m)
            feature_match_loss += self.l1_loss(feat_s, gt_feat_s)
            
            # Normalize by the number of feature maps
            feature_match_loss /= 3.0
            
        # 4. Combine All Losses
        total_loss = (self.w_sl1 * smooth_loss_l1 + 
                      self.w_msssim * msssim_loss + 
                      self.w_drl * c_loss + # Contrast Loss
                      self.w_percep * vgg_percep_loss + # VGG Perceptual Loss
                      self.w_feat_match * feature_match_loss) # Deep Supervision Feature Loss
        
        return total_loss