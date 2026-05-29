# -*- coding: utf-8 -*-
"""
- Cross_Attention + StyleModulation use explicit embed_dim (e.g., 512).
- Self-attention supports:
    (a) Global Self_Attention
    (b) Windowed_Attention (non-shifted)
    (c) Shifted Windowed_Attention (Swin-style roll -> attend -> roll back)
- Shift alternates by block_id (even: 0, odd: ws//2) when use_shifted_wsa=True.
- Residual is added in normalized space (important).

Use:
  restorer = WADNet(channel=16, embed_dim=512, window_size=8,
                    use_windowed_sa=True, use_shifted_wsa=True)

NOTE:
- This shifted version uses roll (wrap-around) WITHOUT an attention mask.
  If you see seam artifacts at borders, add a Swin-style mask later.
"""

from __future__ import absolute_import, division, print_function
import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------
# 1) Utility Functions
# -------------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def window_partition(x, window_size):
    """
    x: (B,C,H,W)
    returns:
      windows: (B*nW, ws, ws, C)
      Hp, Wp padded sizes
    """
    B, C, H, W = x.shape
    Hp = math.ceil(H / window_size) * window_size
    Wp = math.ceil(W / window_size) * window_size
    if Hp != H or Wp != W:
        x = F.pad(x, (0, Wp - W, 0, Hp - H))
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows, Hp, Wp

def window_reverse(windows, window_size, H, W, Hp, Wp):
    """
    windows: (B*nW, ws, ws, C)
    returns: (B,C,H,W) with padding removed
    """
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
    C = windows.shape[3]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
    if Hp != H or Wp != W:
        x = x[:, :, :H, :W]
    return x


# -------------------------
# 2) LayerNorm
# -------------------------
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# -------------------------
# 3) Attention + FFN
# -------------------------
class Cross_Attention(nn.Module):
    """
    Embedding-conditioned channel mixing.
      embedding: (B, embed_dim)
      output: (B, C, H, W)
    """
    def __init__(self, dim, num_heads, bias, embed_dim):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.cph = dim // num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.fc1 = nn.Linear(embed_dim, dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, num_heads * self.cph * self.cph, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
                                   groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, embedding):
        b, c, h, w = x.shape

        q = self.fc2(self.act(self.fc1(embedding))).view(b, self.num_heads, self.cph, self.cph)
        _, v = self.kv_dwconv(self.kv(x)).chunk(2, dim=1)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=self.cph)

        attn = (q * self.temperature).softmax(dim=-1)  # (b, head, cph, cph)
        out = attn @ v                                 # (b, head, cph, hw)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return self.project_out(out)


class Self_Attention(nn.Module):
    """Global OneRestore-style channel attention (C×C per head)."""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (b, head, c, c)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class Windowed_Attention(nn.Module):
    """
    Windowed channel-attention with optional SHIFTED windows (Swin-style roll):
      - shift_size = 0        -> normal non-overlapping windows
      - shift_size = ws//2    -> shift, attend, shift back (cross-window comms via shift)

    NOTE: This is channel-channel attention within each window (OneRestore-style).
    """
    def __init__(self, dim, num_heads, bias, window_size=8, shift_size=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        ss = self.shift_size

        if ws <= 1:
            ss = 0
        if ss > 0:
            # safety: shift must be < window_size
            if ss >= ws:
                raise ValueError(f"shift_size ({ss}) must be < window_size ({ws})")
            x = torch.roll(x, shifts=(-ss, -ss), dims=(2, 3))

        x_windows, Hp, Wp = window_partition(x, ws)             # (B*nW, ws, ws, C)
        x_windows = x_windows.permute(0, 3, 1, 2).contiguous()  # (B*nW, C, ws, ws)

        qkv = self.qkv_dwconv(self.qkv(x_windows))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=ws, w=ws)
        out = self.project_out(out)

        out = window_reverse(out.permute(0, 2, 3, 1), ws, H, W, Hp, Wp)  # (B,C,H,W)

        if ss > 0:
            out = torch.roll(out, shifts=(ss, ss), dims=(2, 3))

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class CALayer(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        r = max(1, dim // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 embed_dim,
                 num_heads=8,
                 ffn_expansion_factor=2.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 use_windowed_sa=True,
                 window_size=8,
                 shift_size=0):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.cross_attn = Cross_Attention(dim, num_heads=num_heads, bias=bias, embed_dim=embed_dim)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        if use_windowed_sa:
            self.self_attn = Windowed_Attention(
                dim, num_heads=num_heads, bias=bias,
                window_size=window_size, shift_size=shift_size
            )
        else:
            self.self_attn = Self_Attention(dim, num_heads=num_heads, bias=bias)

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias=bias)
        self.calayer = CALayer(dim)

    def forward(self, x, embedding):
        x = x + self.cross_attn(self.norm1(x), embedding)
        x = x + self.self_attn(self.norm2(x))
        x = x + self.calayer(self.ffn(self.norm3(x)))
        return x


class ResidualBlock(nn.Module):
    """
    If use_shifted_wsa=True and use_windowed_sa=True:
      - even block_id -> shift = 0
      - odd  block_id -> shift = window_size//2
    """
    def __init__(self,
                 channel,
                 embed_dim,
                 window_size=8,
                 use_windowed_sa=True,
                 use_shifted_wsa=False,
                 block_id=0):
        super().__init__()

        shift = 0
        if use_windowed_sa and use_shifted_wsa:
            shift = (window_size // 2) if (block_id % 2 == 1) else 0

        self.block = TransformerBlock(
            channel,
            embed_dim=embed_dim,
            num_heads=8,
            ffn_expansion_factor=2.0,
            bias=False,
            LayerNorm_type='WithBias',
            use_windowed_sa=use_windowed_sa,
            window_size=window_size,
            shift_size=shift,
        )

    def forward(self, x, embedding):
        return self.block(x, embedding)


# -------------------------
# 4) WADNet Architecture
# -------------------------
class StyleModulation(nn.Module):
    """Embed-conditioned affine transform (gamma, beta)."""
    def __init__(self, c_in, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, c_in * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, embedding):
        gamma_beta = self.proj(embedding)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class RefinementBlock(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.conv_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        )
        self.style_mod = StyleModulation(dim, embed_dim)

    def forward(self, x, embedding):
        refined_feat = self.conv_refine(x)
        modulated_feat = self.style_mod(refined_feat, embedding)
        return x + modulated_feat


class Encoder(nn.Module):
    def __init__(self, channel, embed_dim, window_size=8, use_windowed_sa=True, use_shifted_wsa=False, start_id=0):
        super().__init__()
        # 4 blocks
        self.el  = ResidualBlock(channel,       embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 0)
        self.em  = ResidualBlock(channel * 2,   embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 1)
        self.es  = ResidualBlock(channel * 4,   embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 2)
        self.ess = ResidualBlock(channel * 8,   embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_eltem  = nn.Conv2d(channel,       2 * channel, 1, bias=False)
        self.conv_emtes  = nn.Conv2d(2 * channel,   4 * channel, 1, bias=False)
        self.conv_estess = nn.Conv2d(4 * channel,   8 * channel, 1, bias=False)

    def forward(self, x, embedding):
        elout = self.el(x, embedding)
        x_emin = self.conv_eltem(self.maxpool(elout))
        emout = self.em(x_emin, embedding)
        x_esin = self.conv_emtes(self.maxpool(emout))
        esout = self.es(x_esin, embedding)
        x_essin = self.conv_estess(self.maxpool(esout))
        essout = self.ess(x_essin, embedding)
        return elout, emout, esout, essout


class Backbone(nn.Module):
    def __init__(self, channel, embed_dim, window_size=8, use_windowed_sa=True, use_shifted_wsa=False, start_id=4):
        super().__init__()
        # 2 blocks
        self.s1 = ResidualBlock(channel * 8, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 0)
        self.s2 = ResidualBlock(channel * 8, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 1)

    def forward(self, x, embedding):
        x = self.s1(x, embedding)
        x = self.s2(x, embedding)
        return x


class Decoder(nn.Module):
    def __init__(self, channel, embed_dim, window_size=8, use_windowed_sa=True, use_shifted_wsa=False, start_id=6):
        super().__init__()
        # 4 blocks
        self.dss = ResidualBlock(channel * 8, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 0)
        self.ds  = ResidualBlock(channel * 4, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 1)
        self.dm  = ResidualBlock(channel * 2, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 2)
        self.dl  = ResidualBlock(channel,     embed_dim, window_size, use_windowed_sa, use_shifted_wsa, block_id=start_id + 3)

        self.conv_dsstds = nn.Conv2d(8 * channel, 4 * channel, 1, bias=False)
        self.conv_dstdm  = nn.Conv2d(4 * channel, 2 * channel, 1, bias=False)
        self.conv_dmtdl  = nn.Conv2d(2 * channel, channel,     1, bias=False)

    def _upsample(self, x, y):
        _, _, H0, W0 = y.size()
        return F.interpolate(x, size=(H0, W0), mode='bilinear', align_corners=False)

    def forward(self, x, x_ss, x_s, x_m, x_l, embedding):
        dssout = self.dss(x + x_ss, embedding)
        x_dsin = self.conv_dsstds(self._upsample(dssout, x_s))
        dsout = self.ds(x_dsin + x_s, embedding)
        x_dmin = self.conv_dstdm(self._upsample(dsout, x_m))
        dmout = self.dm(x_dmin + x_m, embedding)
        x_dlin = self.conv_dmtdl(self._upsample(dmout, x_l))
        dlout = self.dl(x_dlin + x_l, embedding)
        return dlout


class WADNet(nn.Module):
    def __init__(self,
                 channel=16,
                 embed_dim=512,
                 window_size=8,
                 use_windowed_sa=True,
                 use_shifted_wsa=False):
        super().__init__()
        self.norm = lambda x: (x - 0.5) / 0.5
        self.denorm = lambda x: (x + 1) / 2

        self.embed_dim = embed_dim
        self.in_conv = nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False)

        # block_id schedule: Encoder(0-3), Backbone(4-5), Decoder(6-9)
        self.encoder = Encoder(channel, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, start_id=0)
        self.middle  = Backbone(channel, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, start_id=4)
        self.decoder = Decoder(channel, embed_dim, window_size, use_windowed_sa, use_shifted_wsa, start_id=6)

        self.refiner = RefinementBlock(channel, embed_dim)
        self.out_conv = nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, embedding):
        x_norm = self.norm(x)
        x_in = self.in_conv(x_norm)

        x_l, x_m, x_s, x_ss = self.encoder(x_in, embedding)
        x_mid = self.middle(x_ss, embedding)
        x_out_feat = self.decoder(x_mid, x_ss, x_s, x_m, x_l, embedding)

        x_refined_feat = self.refiner(x_out_feat, embedding)

        out_norm = self.out_conv(x_refined_feat) + x_norm
        out = self.denorm(out_norm)
        return out, x_l, x_m, x_s


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WADNet(
        channel=16,
        embed_dim=512,
        window_size=8,
        use_windowed_sa=True,
        use_shifted_wsa=True,   # <--- enable alternating shifted windows
    ).to(device)

    x = torch.rand(1, 3, 256, 256).to(device)
    emb = torch.rand(1, 512).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"WADNet (shifted W-SA) total params: {total_params/1e6:.2f} M")

    with torch.no_grad():
        out, xl, xm, xs = model(x, emb)
        print("out:", out.shape, out.min().item(), out.max().item())
