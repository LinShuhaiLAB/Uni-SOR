import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import tifffile
import torchvision.ops as ops
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.ops import roi_align
import pandas as pd
import ast
import math
alpha_train = 0.99
val_break_preview = 999999999
input_pad_size = 8
w_l1 = 2
w_mape = 2
w_mse = 20
w_ms_ssim = 1
w_grid_hist = 10
w_grad = 8
w_grid_contrast = 4
w_extremes = 1
w_freq = 10
w_grad_supp = 0.2
class MSIDataset(Dataset):
    def __init__(self, csv_path, root_dir, cache_data=True):
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.cache_data = cache_data
        self.samples = self._load_from_csv()
        self.image_cache = {}
        if self.cache_data and len(self.samples) > 0:
            print("Pre-loading images into RAM for speed...")
            self._preload_images()
        print(f"Dataset initialized: {len(self.samples)} valid samples.")
    def _load_from_csv(self):
        print(f"Loading dataset mapping from: {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(self.csv_path, encoding='gbk')
        samples = []
        missing_count = 0
        col_to_folder = {
            'HE_Heatmap': 'HE_Heatmap',
            'original': 'original',
            'sampling': 'sampling'
        }
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying paths"):
            try:
                def get_real_path(col_name):
                    raw_path = str(row[col_name])
                    filename = os.path.basename(raw_path.replace('\\', '/'))
                    return os.path.join(self.root_dir, col_to_folder[col_name], filename)
                p_heatmap = get_real_path('HE_Heatmap')
                p_gt = get_real_path('original')
                p_input = get_real_path('sampling')
                if not (os.path.exists(p_heatmap) and os.path.exists(p_gt) and os.path.exists(p_input)):
                    missing_count += 1
                    continue
                grid_y = ast.literal_eval(row['grid_y'])
                grid_x = ast.literal_eval(row['grid_x'])
                samples.append({
                    'heatmap': p_heatmap,
                    'gt': p_gt,
                    'input': p_input,
                    'name': str(row['ID']),
                    'grid_lines': (grid_y, grid_x)
                })
            except Exception:
                missing_count += 1
                continue
        if missing_count > 0:
            print(f"Note: Skipped {missing_count} samples due to missing files.")
        return samples
    def _load_tiff(self, path, force_gray=False, return_stats=False):
        try:
            path = str(path)
            if not os.path.exists(path):
                out = torch.zeros(1 if force_gray else 3, 256, 256)
                if return_stats:
                    return out, {
                        "raw_min": None, "raw_max": None,
                        "norm_min": None, "norm_max": None,
                        "valid_count": 0
                    }
                return out
            img = tifffile.imread(path).astype(np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            alpha = None
            if img.ndim == 3:
                if img.shape[2] == 4:
                    alpha = img[:, :, 3]
                    rgb = img[:, :, :3]
                    img = rgb
                elif img.shape[0] == 4 and img.shape[1] > 4 and img.shape[2] > 4:
                    alpha = img[3, :, :]
                    rgb = img[:3, :, :]
                    img = rgb
            if alpha is None:
                if img.ndim == 2:
                    valid_mask = np.ones_like(img, dtype=bool)
                elif img.ndim == 3:
                    if img.shape[0] in (1, 3) and img.shape[0] < img.shape[-1]:
                        valid_mask = np.ones((img.shape[1], img.shape[2]), dtype=bool)
                    else:
                        valid_mask = np.ones((img.shape[0], img.shape[1]), dtype=bool)
            else:
                valid_mask = (alpha > 0)
            if force_gray:
                if img.ndim == 3:
                    if img.shape[2] >= 3:
                        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                    elif img.shape[2] == 1:
                        img = img[:, :, 0]
                    elif img.shape[0] >= 3:
                        img = 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
                    elif img.shape[0] == 1:
                        img = img[0, :, :]
            if img.ndim == 2:
                vals = img[valid_mask]
            else:
                if img.shape[-1] in (1, 3):
                    vals = img[valid_mask, :].reshape(-1)
                elif img.shape[0] in (1, 3) and img.shape[0] < img.shape[-1]:
                    vals = img[:, valid_mask].reshape(-1)
                else:
                    vals = img.reshape(-1)
            valid_count = int(vals.size)
            if valid_count == 0:
                raw_min = raw_max = None
            else:
                raw_min = float(np.min(vals))
                raw_max = float(np.max(vals))
            if raw_max is not None and raw_max > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
            if img.ndim == 2:
                nvals = img[valid_mask]
            else:
                if img.shape[-1] in (1, 3):
                    nvals = img[valid_mask, :].reshape(-1)
                elif img.shape[0] in (1, 3) and img.shape[0] < img.shape[-1]:
                    nvals = img[:, valid_mask].reshape(-1)
                else:
                    nvals = img.reshape(-1)
            if valid_count == 0:
                norm_min = norm_max = None
            else:
                norm_min = float(np.min(nvals))
                norm_max = float(np.max(nvals))
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim == 3 and img.shape[2] < img.shape[0]:
                img = img.transpose(2, 0, 1)
            out = torch.from_numpy(np.ascontiguousarray(np.clip(img, 0.0, 1.0)))
            if return_stats:
                return out, {
                    "raw_min": raw_min, "raw_max": raw_max,
                    "norm_min": norm_min, "norm_max": norm_max,
                    "valid_count": valid_count
                }
            return out
        except Exception:
            out = torch.zeros(1 if force_gray else 3, 256, 256)
            if return_stats:
                return out, {
                    "raw_min": None, "raw_max": None,
                    "norm_min": None, "norm_max": None,
                    "valid_count": 0
                }
            return out
    def _preload_images(self):
        stat = {
            "input": {"raw_min": 1e30, "raw_max": -1e30, "norm_min": 1e30, "norm_max": -1e30, "valid_count": 0},
            "gt": {"raw_min": 1e30, "raw_max": -1e30, "norm_min": 1e30, "norm_max": -1e30, "valid_count": 0},
        }
        for idx, s in enumerate(tqdm(self.samples, desc="Caching Data")):
            inp, st_inp = self._load_tiff(s['input'], True, return_stats=True)
            he = self._load_tiff(s['heatmap'], False)
            gt, st_gt = self._load_tiff(s['gt'], True, return_stats=True)
            self.image_cache[idx] = (inp, he, gt)
            if st_inp["valid_count"] > 0:
                stat["input"]["raw_min"] = min(stat["input"]["raw_min"], st_inp["raw_min"])
                stat["input"]["raw_max"] = max(stat["input"]["raw_max"], st_inp["raw_max"])
                stat["input"]["norm_min"] = min(stat["input"]["norm_min"], st_inp["norm_min"])
                stat["input"]["norm_max"] = max(stat["input"]["norm_max"], st_inp["norm_max"])
                stat["input"]["valid_count"] += st_inp["valid_count"]
            if st_gt["valid_count"] > 0:
                stat["gt"]["raw_min"] = min(stat["gt"]["raw_min"], st_gt["raw_min"])
                stat["gt"]["raw_max"] = max(stat["gt"]["raw_max"], st_gt["raw_max"])
                stat["gt"]["norm_min"] = min(stat["gt"]["norm_min"], st_gt["norm_min"])
                stat["gt"]["norm_max"] = max(stat["gt"]["norm_max"], st_gt["norm_max"])
                stat["gt"]["valid_count"] += st_gt["valid_count"]
        print("[Input/GT Range Preview | ignore transparent]")
        for k in ["input", "gt"]:
            if stat[k]["valid_count"] == 0:
                print(f" - {k}: no valid pixels found (alpha==0 everywhere?)")
            else:
                print(
                    f" - {k}: raw_min={stat[k]['raw_min']:.4f}, raw_max={stat[k]['raw_max']:.4f} | "
                    f"norm_min={stat[k]['norm_min']:.4f}, norm_max={stat[k]['norm_max']:.4f} | "
                    f"valid_pixels={stat[k]['valid_count']}"
                )
    def _compute_boxes(self, H, W, h_lines, v_lines):
        ys = sorted(list(set([0] + h_lines + [H])))
        xs = sorted(list(set([0] + v_lines + [W])))
        boxes = []
        for r in range(len(ys) - 1):
            for c in range(len(xs) - 1):
                y0, y1 = ys[r], ys[r + 1]
                x0, x1 = xs[c], xs[c + 1]
                if (y1 - y0) < 2 or (x1 - x0) < 2: continue
                boxes.append([float(x0), float(y0), float(x1), float(y1)])
        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.cache_data and idx in self.image_cache:
            inp, he, gt = self.image_cache[idx]
        else:
            inp = self._load_tiff(s['input'], True)
            he = self._load_tiff(s['heatmap'], False)
            gt = self._load_tiff(s['gt'], True)
        h_lines = s['grid_lines'][0]
        v_lines = s['grid_lines'][1]
        H, W = inp.shape[1], inp.shape[2]
        boxes = self._compute_boxes(H, W, h_lines, v_lines)
        return {
            'input': inp,
            'he': he,
            'gt': gt,
            'boxes': boxes,
            'h_lines': h_lines, 'v_lines': v_lines,
            'name': s['name']
        }
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GroupNorm(16, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GroupNorm(16, dim)
        )
    def forward(self, x):
        return x + self.block(x)
class DynamicDepthBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = ResBlock(dim)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.gate(x).view(x.size(0), 1, 1, 1)
        return x + w * self.body(x)
class EfficientCrossScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=4, reduction_ratio=2, global_dim=512):
        super().__init__()
        self.num_heads = num_heads
        reduced_dim = dim // reduction_ratio
        self.reduce_local = nn.Conv2d(dim, reduced_dim, 1)
        self.proj_global = nn.Conv2d(global_dim, reduced_dim, 1)
        self.attn = nn.MultiheadAttention(embed_dim=reduced_dim, num_heads=num_heads, batch_first=True)
        self.restore_local = nn.Conv2d(reduced_dim, dim, 1)
        self.norm_reduced = nn.LayerNorm(reduced_dim)
    def forward(self, x_local, x_global, batch_indices):
        N, C, H, W = x_local.shape
        local_feat = self.reduce_local(x_local)
        q = local_feat.flatten(2).transpose(1, 2)
        q = self.norm_reduced(q)
        global_feat = self.proj_global(x_global)
        k_v_flat = global_feat.flatten(2).transpose(1, 2)
        k_v_flat = self.norm_reduced(k_v_flat)
        curr_kv = k_v_flat[batch_indices]
        attn_out, _ = self.attn(query=q, key=curr_kv, value=curr_kv)
        out = attn_out.transpose(1, 2).reshape(N, -1, H, W)
        out = self.restore_local(out)
        return out
class BlockCorrelation(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x, batch_indices):
        N, C, H, W = x.shape
        if N <= 1: return x
        output = x.clone()
        unique_batches = torch.unique(batch_indices)
        for b in unique_batches:
            mask = (batch_indices == b)
            indices = torch.where(mask)[0]
            if len(indices) <= 1:
                continue
            x_batch = x[mask]
            k = x_batch.shape[0]
            feat_batch = x_batch.mean(dim=[2, 3])
            feat_in = feat_batch.unsqueeze(0)
            feat_norm = self.norm(feat_in)
            attn_out, _ = self.attn(feat_norm, feat_norm, feat_norm)
            delta = attn_out.squeeze(0).contiguous().view(k, C, 1, 1)
            res_val = x_batch + self.gamma * delta
            output[indices] = res_val.to(output.dtype)
        return output
class GradientSurgeGate(nn.Module):
    """
    梯度激增门控模块：
    专门用于捕捉和增强相邻区域之间的数值激增或骤减。
    """
    def __init__(self, dim):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim + 1, dim // 4, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.shape
        x_avg = x.mean(dim=1, keepdim=True)
        g_x = F.conv2d(x_avg, self.sobel_x, padding=1)
        g_y = F.conv2d(x_avg, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(g_x ** 2 + g_y ** 2 + 1e-8)
        combined = torch.cat([x, grad_mag], dim=1)
        gate = self.gate_net(combined)
        out = x + self.scale * (x * gate * (1.0 + grad_mag))
        return out
class PatchDiscriminator(nn.Module):
    """
    PatchGAN 判别器：
    输出不是一个单一的真假值，而是一个 N x N 的矩阵，
    其中每个点代表原图中一块区域 (Patch) 的真假。
    """
    def __init__(self, in_channels=1, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        return self.model(input)
class FeatureAlignmentModule(nn.Module):
    def __init__(self, in_channels_he, in_channels_msi, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels_he + in_channels_msi, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, 2 * kernel_size * kernel_size, 3, 1, 1)
        )
        nn.init.constant_(self.offset_conv[2].weight, 0)
        nn.init.constant_(self.offset_conv[2].bias, 0)
        self.modulator_conv = nn.Sequential(
            nn.Conv2d(in_channels_he + in_channels_msi, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, kernel_size * kernel_size, 3, 1, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.modulator_conv[2].bias, -2.0)
        self.he_conv = nn.Conv2d(in_channels_he, out_channels, kernel_size, 1, self.padding)
    def forward(self, he_feat, msi_feat):
        combined = torch.cat([he_feat, msi_feat], dim=1)
        offsets = self.offset_conv(combined)
        mask = self.modulator_conv(combined)
        aligned_he = ops.deform_conv2d(
            input=he_feat,
            offset=offsets,
            weight=self.he_conv.weight,
            bias=self.he_conv.bias,
            padding=self.padding,
            mask=mask
        )
        return aligned_he
class LearnableDeblurKernel(nn.Module):
    def __init__(self, in_channels=1, kernel_size=1, hidden_dim=64):
        super().__init__()
        self.kernel_size = kernel_size
        self.K = kernel_size
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, kernel_size * kernel_size, 1, 1, 0),
        )
        nn.init.constant_(self.weight_generator[-1].bias, 1.0 / (kernel_size * kernel_size))
        nn.init.normal_(self.weight_generator[-1].weight, 0, 0.01)
    def forward(self, x_coarse):
        N, C, H, W = x_coarse.shape
        K = self.K
        weights = self.weight_generator(x_coarse)
        weights = F.softmax(weights, dim=1)
        weights = weights.view(N, K, K, H, W)
        weights = weights.permute(0, 3, 4, 1, 2).contiguous()
        x_expand = x_coarse.unsqueeze(-1).unsqueeze(-1)
        weights = weights.unsqueeze(1)
        out = x_expand * weights
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous()
        out = out.view(N, C, H * K, W * K)
        return out
class DeblurUpsampleModule(nn.Module):
    def __init__(self, in_channels=1, kernel_size=1, use_residual=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.deblur = LearnableDeblurKernel(in_channels, kernel_size, hidden_dim=64)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, in_channels, 3, 1, 1),
        )
    def forward(self, x_coarse):
        x_fine = self.deblur(x_coarse)
        if self.use_residual:
            residual = self.refine(x_fine)
            x_fine = x_fine + residual
        else:
            x_fine = self.refine(x_fine)
        return torch.clamp(x_fine, 0.0, 1.0)
class SingleStageBlockNet(nn.Module):
    def __init__(self, input_channels=1, he_channels=3, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels + he_channels, 32, 3, 1, 1),
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.GroupNorm(16, embed_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block_correlation = BlockCorrelation(embed_dim)
        self.global_pool_reducer = nn.AdaptiveAvgPool2d((32, 32))
        self.cross_attention = EfficientCrossScaleAttention(embed_dim, reduction_ratio=2)
        self.context_fusion = nn.Sequential(
            DynamicDepthBlock(embed_dim),
            DynamicDepthBlock(embed_dim)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, 1, 1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.skip_fuse1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            DynamicDepthBlock(64)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.skip_fuse2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            DynamicDepthBlock(32)
        )
        self.surge_gate = GradientSurgeGate(32)
        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)
    def forward(self, patches, global_raw_feat, batch_indices):
        N_patches = patches.shape[0]
        if N_patches == 0: return patches[:, 0:1, :, :]
        f1 = self.enc1(patches)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f3 = self.block_correlation(f3, batch_indices)
        global_spatial = self.global_pool_reducer(global_raw_feat)
        attn_out = self.cross_attention(f3, global_spatial, batch_indices)
        fusion_feat = self.context_fusion(f3 + attn_out)
        d1 = self.up1(fusion_feat)
        d1_cat = torch.cat([d1, f2], dim=1)
        d1_fused = self.skip_fuse1(d1_cat)
        d2 = self.up2(d1_fused)
        d2_cat = torch.cat([d2, f1], dim=1)
        d2_fused = self.skip_fuse2(d2_cat)
        enhanced_feat = self.surge_gate(d2_fused)
        out = self.final_conv(enhanced_feat)
        return out
def apply_grid_median_smoothing(img, h_lines_list, v_lines_list):
    """向量化的网格中位数平滑"""
    B, C, H, W = img.shape
    blocky_img = img.clone()
    for b in range(B):
        ys = sorted(list(set([0] + list(h_lines_list[b]) + [H])))
        xs = sorted(list(set([0] + list(v_lines_list[b]) + [W])))
        grid_bounds = []
        for r in range(len(ys) - 1):
            for c in range(len(xs) - 1):
                y1, y2 = int(ys[r]), int(ys[r + 1])
                x1, x2 = int(xs[c]), int(xs[c + 1])
                if y2 > y1 and x2 > x1:
                    grid_bounds.append((y1, y2, x1, x2))
        if not grid_bounds: continue
        boxes = torch.tensor([[x1, y1, x2, y2] for y1, y2, x1, x2 in grid_bounds], device=img.device, dtype=img.dtype)
        idx_col = torch.zeros((len(boxes), 1), device=img.device, dtype=img.dtype)
        rois = torch.cat([idx_col, boxes], dim=1)
        img_b = img[b:b + 1]
        max_h = max(y2 - y1 for y1, y2, x1, x2 in grid_bounds)
        max_w = max(x2 - x1 for y1, y2, x1, x2 in grid_bounds)
        max_h = max(1, int(max_h))
        max_w = max(1, int(max_w))
        patches = roi_align(img_b, rois, output_size=(max_h, max_w), spatial_scale=1.0, sampling_ratio=4)
        for i, (y1, y2, x1, x2) in enumerate(grid_bounds):
            patch = patches[i, :, :y2 - y1, :x2 - x1]
            median_val = patch.reshape(C, -1).median(dim=1)[0].view(C, 1, 1)
            blocky_img[b, :, y1:y2, x1:x2] = median_val
    return blocky_img
class CascadeInpaintingNet(nn.Module):
    def __init__(self, msi_channels=1, he_channels=3):
        super().__init__()
        self.he_encoder = nn.Sequential(
            nn.Conv2d(he_channels, 64, 3, 2, 1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.GroupNorm(16, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.align_module = FeatureAlignmentModule(in_channels_he=he_channels, in_channels_msi=msi_channels,
                                                   out_channels=32)
        self.stage1 = SingleStageBlockNet(input_channels=msi_channels, he_channels=32, embed_dim=128)
        self.stage2 = SingleStageBlockNet(input_channels=msi_channels * 2, he_channels=32, embed_dim=128)
        nn.init.constant_(self.stage2.final_conv.weight, 0.0)
        nn.init.constant_(self.stage2.final_conv.bias, 0.0)
        self.context_ratio = 0.15
        self.deconv_refine = DeblurUpsampleModule(in_channels=1, kernel_size=1, use_residual=True)
    def _expand_boxes(self, boxes, H, W):
        expanded = boxes.clone()
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        pad_w = widths * self.context_ratio
        pad_h = heights * self.context_ratio
        expanded[:, 0] = torch.clamp(boxes[:, 0] - pad_w, 0, W)
        expanded[:, 1] = torch.clamp(boxes[:, 1] - pad_h, 0, H)
        expanded[:, 2] = torch.clamp(boxes[:, 2] + pad_w, 0, W)
        expanded[:, 3] = torch.clamp(boxes[:, 3] + pad_h, 0, H)
        return expanded
    def _reconstruct_images(self, batch_size, channels, H, W, patches, boxes, batch_indices, device):
        outputs_sum = torch.zeros(batch_size, channels, H, W, device=device)
        counts_map = torch.zeros(batch_size, 1, H, W, device=device)
        _, ph, pw = patches[0].shape if len(patches) > 0 else (1, 1, 1)
        h_crop = int(ph * (self.context_ratio / (1 + 2 * self.context_ratio)))
        w_crop = int(pw * (self.context_ratio / (1 + 2 * self.context_ratio)))
        use_crop = h_crop > 0 and w_crop > 0 and (ph - 2 * h_crop) > 2 and (pw - 2 * w_crop) > 2
        if use_crop:
            patches_center = patches[:, :, h_crop:-h_crop, w_crop:-w_crop]
        else:
            patches_center = patches
        for b_idx in range(batch_size):
            mask = (batch_indices == b_idx)
            if not mask.any():
                continue
            b_patches = patches_center[mask]
            b_boxes = boxes[mask].int()
            for i in range(len(b_boxes)):
                x1, y1, x2, y2 = b_boxes[i].tolist()
                target_h, target_w = y2 - y1, x2 - x1
                if target_h <= 0 or target_w <= 0:
                    continue
                patch_resized = F.interpolate(
                    b_patches[i:i + 1], size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                outputs_sum[b_idx, :, y1:y2, x1:x2] += patch_resized
                counts_map[b_idx, :, y1:y2, x1:x2] += 1.0
        return outputs_sum / counts_map.clamp(min=1.0)
    def forward(self, input_imgs, he_imgs, h_lines_list, v_lines_list, roi_boxes_list):
        device = input_imgs.device
        batch_size, _, H, W = input_imgs.shape
        global_feat = self.he_encoder(he_imgs)
        all_rois = []
        all_expanded_rois = []
        batch_indices_list = []
        for b_idx, boxes in enumerate(roi_boxes_list):
            if len(boxes) > 0:
                idx_col = torch.full((len(boxes), 1), b_idx, device=device)
                expanded_boxes = self._expand_boxes(boxes, H, W)
                rois_expanded_with_idx = torch.cat([idx_col, expanded_boxes], dim=1)
                all_rois.append(boxes)
                all_expanded_rois.append(rois_expanded_with_idx)
                batch_indices_list.append(torch.full((len(boxes),), b_idx, dtype=torch.long, device=device))
        if not all_rois:
            return torch.zeros_like(input_imgs[:, 0:1]), torch.zeros_like(input_imgs[:, 0:1])
        cat_expanded_rois = torch.cat(all_expanded_rois, dim=0)
        cat_batch_indices = torch.cat(batch_indices_list, dim=0)
        cat_original_boxes = torch.cat(all_rois, dim=0)
        pad_size = input_pad_size
        patches_input = roi_align(input_imgs, cat_expanded_rois, output_size=(pad_size, pad_size), spatial_scale=1.0, sampling_ratio=4)
        patches_he = roi_align(he_imgs, cat_expanded_rois, output_size=(pad_size, pad_size), spatial_scale=1.0, sampling_ratio=4)
        norm_input = patches_input
        aligned_he_feat = self.align_module(patches_he, norm_input)
        stage1_in = torch.cat([norm_input, aligned_he_feat], dim=1)
        patches_coarse = self.stage1(stage1_in, global_feat, cat_batch_indices)
        patches_coarse_clamped = torch.clamp(patches_coarse, 0.0, 1.0)
        stage2_in = torch.cat([patches_coarse_clamped, norm_input, aligned_he_feat], dim=1)
        patches_residual = self.stage2(stage2_in, global_feat, cat_batch_indices)
        patches_final = patches_coarse_clamped + patches_residual + norm_input
        patches_final = torch.clamp(patches_final, 0.0, 1.0)
        final_img_pixel = self._reconstruct_images(batch_size, 1, H, W, patches_final, cat_original_boxes,
                                                   cat_batch_indices, device)
        coarse_img = self._reconstruct_images(batch_size, 1, H, W, patches_coarse_clamped, cat_original_boxes,
                                              cat_batch_indices, device)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            final_img_pixel = self.deconv_refine(final_img_pixel.float())
            coarse_img = self.deconv_refine(coarse_img.float())
        final_img_pixel = final_img_pixel.to(input_imgs.dtype)
        coarse_img = coarse_img.to(input_imgs.dtype)
        final_img_blocky = apply_grid_median_smoothing(final_img_pixel, h_lines_list, v_lines_list)
        return final_img_blocky, coarse_img
class SSIM(nn.Module):
    def __init__(self, window_size=81, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        window = self.create_window(window_size, channel)
        self.register_buffer('window', window)
    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    def _ssim(self, img1, img2, window, window_size, channel):
        img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=0.0)
        img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=0.0)
        img1 = torch.clamp(img1, 0.0, 1.0)
        img2 = torch.clamp(img2, 0.0, 1.0)
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / (denominator + 1e-8)
        ssim_map = torch.nan_to_num(ssim_map, nan=0.0, posinf=1.0, neginf=0.0)
        ssim_map = torch.clamp(ssim_map, -1.0, 1.0)
        return ssim_map
    def forward(self, img1, img2):
        return self._ssim(img1, img2, self.window, self.window_size, self.channel)
class MacroGridGradientLoss(nn.Module):
    def __init__(self, edge_k=20.0, missing_k=5.0, w_cap=50.0):
        super().__init__()
        self.edge_k = edge_k
        self.missing_k = missing_k
        self.w_cap = w_cap
    def forward(self, pred, gt, valid_mask, boxes_list):
        device = pred.device
        rois = []
        for b_idx, boxes in enumerate(boxes_list):
            if boxes is None or len(boxes) == 0:
                continue
            boxes = boxes.to(device)
            idx_col = torch.full((len(boxes), 1), b_idx, device=device, dtype=boxes.dtype)
            rois.append(torch.cat([idx_col, boxes], dim=1))
        if len(rois) == 0:
            return pred.new_tensor(0.0)
        rois_cat = torch.cat(rois, dim=0)
        pred_sum = roi_align(pred * valid_mask, rois_cat, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1)
        gt_sum = roi_align(gt * valid_mask, rois_cat, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1)
        m_sum = roi_align(valid_mask, rois_cat, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1).clamp_min(1e-6)
        pred_mean = pred_sum / m_sum
        gt_mean = gt_sum / m_sum
        pairs_i, pairs_j = [], []
        cursor = 0
        for boxes in boxes_list:
            k = 0 if (boxes is None) else len(boxes)
            if k == 0:
                continue
            bcpu = boxes.detach().round().long().cpu()
            xs0 = bcpu[:, 0].tolist();
            ys0 = bcpu[:, 1].tolist()
            xs1 = bcpu[:, 2].tolist();
            ys1 = bcpu[:, 3].tolist()
            loc = {(xs0[t], ys0[t]): (cursor + t) for t in range(k)}
            for t in range(k):
                rk = (xs1[t], ys0[t])
                if rk in loc:
                    pairs_i.append(cursor + t);
                    pairs_j.append(loc[rk])
                dk = (xs0[t], ys1[t])
                if dk in loc:
                    pairs_i.append(cursor + t);
                    pairs_j.append(loc[dk])
            cursor += k
        if len(pairs_i) == 0:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                p = pred.float()
                g = gt.float()
                m = valid_mask.float()
                kx = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]], device=device).view(1, 1, 3, 3) / 4.0
                ky = torch.tensor([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]], device=device).view(1, 1, 3, 3) / 4.0
                total = p.new_tensor(0.0)
                sizes = [(38, 38), (76, 76)]
                for size in sizes:
                    p_num = F.adaptive_avg_pool2d(p * m, size)
                    g_num = F.adaptive_avg_pool2d(g * m, size)
                    m_s = F.adaptive_avg_pool2d(m, size).clamp_min(1e-6)
                    p_s = p_num / m_s
                    g_s = g_num / m_s
                    pgx = F.conv2d(p_s, kx, padding=1)
                    pgy = F.conv2d(p_s, ky, padding=1)
                    ggx = F.conv2d(g_s, kx, padding=1)
                    ggy = F.conv2d(g_s, ky, padding=1)
                    pgrad = torch.sqrt(pgx ** 2 + pgy ** 2 + 1e-6)
                    ggrad = torch.sqrt(ggx ** 2 + ggy ** 2 + 1e-6)
                    w = 1.0 + self.edge_k * (ggrad ** 2)
                    w = torch.clamp(w, max=self.w_cap)
                    missing = F.relu(ggrad - pgrad) * self.missing_k
                    loss_map = (pgrad - ggrad).abs() + missing
                    total = total + (loss_map * w * m_s).sum() / (m_s.sum() + 1.0)
                return total / len(sizes)
        pi = torch.tensor(pairs_i, device=device, dtype=torch.long)
        pj = torch.tensor(pairs_j, device=device, dtype=torch.long)
        pred_grad = (pred_mean[pi] - pred_mean[pj]).abs()
        gt_grad = (gt_mean[pi] - gt_mean[pj]).abs()
        w = 1.0 + self.edge_k * (gt_grad ** 2)
        w = torch.clamp(w, max=self.w_cap).detach()
        missing = F.relu(gt_grad - pred_grad) * self.missing_k
        loss = ((pred_grad - gt_grad).abs() + missing) * w
        return loss.mean()
class BlockAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.ssim_windows = [15, 31, 47]
        self.ssim_modules = nn.ModuleList([SSIM(window_size=ws, channel=1) for ws in self.ssim_windows])
        self.ssim_weights = [0.5, 0.3, 0.2]
        self.hist_bins = 32
        self.hist_sigma = 0.05
        self.hist_down = 128
        self.grad_loss = MacroGridGradientLoss(edge_k=3, missing_k=3.0, w_cap=50.0)
        self.mape_cap = 10.0
        self.mape_denom_eps = 1e-8
        self.progress = 0.0
        self.peak_preserve_w = 0.3
        self.peak_pool_ks = 3
        self.peak_delta_ratio = 0.15
        self.peak_delta_min = 0.03
        self.peak_delta_max = 0.20
        self.peak_w_scale = 4.0
        self.extrema_pixel_boost_peak = 1.5
        self.extrema_pixel_boost_valley = 1.5
        self.freq_size = 128
        wf = self.freq_size // 2 + 1
        yy = torch.linspace(0.0, 1.0, self.freq_size).view(self.freq_size, 1)
        xx = torch.linspace(0.0, 1.0, wf).view(1, wf)
        rr = torch.sqrt(yy * yy + xx * xx)
        self.register_buffer('freq_weight', rr.pow(2.0))
        self.gradient_suppress_k = 0.25  # 可调节的k值，表示前k%
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x_suppress', sobel_x)
        self.register_buffer('sobel_y_suppress', sobel_y)
        self.loss_stats = {
            'loss_l1': [],
            'loss_mape': [],
            'loss_mse': [],
            'loss_ms_ssim': [],
            'loss_grid_hist': [],
            'loss_grad': [],
            'loss_grid_contrast': [],
            'loss_peak_under': [],
            'loss_valley_over': [],
            'loss_extremes': [],
            'loss_freq': [],
            'loss_grad_supp': [],
            'total': []
        }
    def _soft_hist(self, x, bins=32, sigma=0.03):
        x = x.float().clamp(0.0, 1.0)
        centers = torch.linspace(0.0, 1.0, bins, device=x.device, dtype=torch.float32)
        diff = (x.unsqueeze(1) - centers.unsqueeze(0)) / sigma
        diff = diff.clamp(-20.0, 20.0)
        w = torch.exp(-0.5 * diff ** 2)
        h = w.sum(dim=0) + 1e-8
        return h / h.sum()
    def set_progress(self, progress: float):
        self.progress = float(progress)
    def _compute_gradient_suppress_loss(self, pred, gt, input_img, valid_mask, k_percent=None):
        if k_percent is None:
            k_percent = self.gradient_suppress_k
        if input_img is None or valid_mask.sum() < 10:
            return pred.new_tensor(0.0)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            input_f = input_img.float()
            pred_f = pred.float()
            mask_f = valid_mask.float()
            input_gx = F.conv2d(input_f, self.sobel_x_suppress, padding=1)
            input_gy = F.conv2d(input_f, self.sobel_y_suppress, padding=1)
            input_grad_mag = torch.sqrt(input_gx ** 2 + input_gy ** 2 + 1e-8)
            pred_gx = F.conv2d(pred_f, self.sobel_x_suppress, padding=1)
            pred_gy = F.conv2d(pred_f, self.sobel_y_suppress, padding=1)
            pred_grad_mag = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)
            input_grad_flat = input_grad_mag.view(input_grad_mag.shape[0], -1)
            k_index = max(1, int(input_grad_flat.shape[1] * k_percent))
            threshold_values = torch.topk(input_grad_flat, k_index, dim=1)[0][:, -1:]
            threshold_map = threshold_values.view(-1, 1, 1, 1)
            input_strong_grad_mask = (input_grad_mag >= threshold_map).float()
            pred_grad_flat = pred_grad_mag.view(pred_grad_mag.shape[0], -1)
            loss_total = pred_f.new_tensor(0.0)
            for b in range(pred_f.shape[0]):
                strong_positions = input_strong_grad_mask[b].view(-1)
                pred_grads_at_strong = pred_grad_flat[b][strong_positions > 0.5]
                if pred_grads_at_strong.numel() == 0:
                    continue
                all_pred_grads = pred_grad_flat[b]
                pred_k_index = max(1, int(all_pred_grads.shape[0] * k_percent))
                pred_threshold = torch.topk(all_pred_grads, pred_k_index)[0][-1]
                violation = F.relu(pred_grads_at_strong - pred_threshold)
                loss_total = loss_total + violation.mean()
            loss_total = loss_total / max(1, pred_f.shape[0])
        return torch.nan_to_num(loss_total, nan=0.0, posinf=10.0, neginf=0.0)
    def forward(self, pred, gt, boxes_list=None, input_img=None):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        bg_mask = (gt <= 1e-6).float()
        valid_mask = (gt > 1e-6).float()
        num_valid = valid_mask.sum()
        if num_valid < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred_u, gt_u = pred, gt
        B, _, H, W = gt_u.shape
        with torch.no_grad():
            gt_valid = gt_u[valid_mask > 0.5]
            if gt_valid.numel() > 0:
                norm_min = gt_valid.min()
                norm_max = gt_valid.max()
            else:
                norm_min = gt_u.new_tensor(0.0)
                norm_max = gt_u.new_tensor(1.0)
        high_th = torch.clamp(0.85 * norm_max, 0.0, 1.0)
        low_th = torch.clamp(0.25 * norm_max, 0.0, 1.0)
        base_weight = 1.0 + (gt_u * 5.0)
        high_val_mask = (gt_u > high_th).float()
        low_val_mask = (gt_u < low_th).float()
        pixel_weight = (base_weight + ((high_val_mask + low_val_mask) * 2.0)) * valid_mask
        boost_map = gt_u.new_ones((B, 1, H, W), dtype=torch.float32)
        peak_cell_mask_map = gt_u.new_zeros((B, 1, H, W), dtype=torch.float32)
        valley_cell_mask_map = gt_u.new_zeros((B, 1, H, W), dtype=torch.float32)
        if boxes_list is not None:
            with torch.no_grad():
                dyn_delta = torch.clamp(
                    (norm_max - norm_min) * self.peak_delta_ratio,
                    min=self.peak_delta_min, max=self.peak_delta_max
                ).float()
                peak_boost = float(1.0 + self.extrema_pixel_boost_peak)
                valley_boost = float(1.0 + self.extrema_pixel_boost_valley)
                device = pred.device
                for b_idx, boxes in enumerate(boxes_list):
                    if boxes is None or len(boxes) == 0:
                        continue
                    boxes = boxes.to(device)
                    idx_col = torch.full((len(boxes), 1), b_idx, device=device, dtype=boxes.dtype)
                    rois = torch.cat([idx_col, boxes], dim=1)
                    gt_sum = roi_align(gt_u * valid_mask, rois, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1)
                    m_sum = roi_align(valid_mask, rois, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1).clamp_min(1e-6)
                    gt_mean = (gt_sum / m_sum).float()
                    bcpu = boxes.detach().round().long().cpu()
                    xs0 = bcpu[:, 0].tolist();
                    ys0 = bcpu[:, 1].tolist()
                    xs1 = bcpu[:, 2].tolist();
                    ys1 = bcpu[:, 3].tolist()
                    K = len(xs0)
                    loc = {(xs0[t], ys0[t]): t for t in range(K)}
                    neigh = [[] for _ in range(K)]
                    for t in range(K):
                        rk = (xs1[t], ys0[t])
                        if rk in loc:
                            j = loc[rk]
                            neigh[t].append(j);
                            neigh[j].append(t)
                        dk = (xs0[t], ys1[t])
                        if dk in loc:
                            j = loc[dk]
                            neigh[t].append(j);
                            neigh[j].append(t)
                    if all(len(neigh[t]) == 0 for t in range(K)):
                        continue
                    neigh_mean = gt_mean.new_empty((K,))
                    for t in range(K):
                        if len(neigh[t]) == 0:
                            neigh_mean[t] = gt_mean[t]
                        else:
                            idx = torch.tensor(neigh[t], device=device, dtype=torch.long)
                            neigh_mean[t] = gt_mean[idx].mean()
                    peak_grid = (gt_mean > (neigh_mean + dyn_delta)) & (gt_mean > high_th.float())
                    valley_grid = (gt_mean < (neigh_mean - dyn_delta)) & (gt_mean < low_th.float())
                    for t in range(K):
                        x0, y0, x1, y1 = xs0[t], ys0[t], xs1[t], ys1[t]
                        x0 = max(0, min(x0, W));
                        x1 = max(0, min(x1, W))
                        y0 = max(0, min(y0, H));
                        y1 = max(0, min(y1, H))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        if bool(peak_grid[t].item()):
                            boost_map[b_idx, 0, y0:y1, x0:x1] *= peak_boost
                            peak_cell_mask_map[b_idx, 0, y0:y1, x0:x1] = 1.0
                        if bool(valley_grid[t].item()):
                            boost_map[b_idx, 0, y0:y1, x0:x1] *= valley_boost
                            valley_cell_mask_map[b_idx, 0, y0:y1, x0:x1] = 1.0
        pixel_weight = (pixel_weight * boost_map.to(pixel_weight.dtype)).clamp(0.0, 5.0)
        loss_l1 = (self.l1(pred_u, gt_u) * pixel_weight).sum() / (num_valid + 1e-8)
        loss_l1_bg = (self.l1(pred, gt) * bg_mask).sum() / (bg_mask.sum() + 1e-8)
        loss_l1 = loss_l1 + 0.1 * loss_l1_bg
        loss_mse = (self.mse(pred_u, gt_u) * pixel_weight).sum() / (num_valid + 1e-8)
        loss_ms_ssim = pred_u.new_tensor(0.0)
        for ssim_module, weight in zip(self.ssim_modules, self.ssim_weights):
            ssim_map = torch.nan_to_num(ssim_module(pred_u, gt_u), nan=0.0, posinf=1.0, neginf=0.0)
            ssim_value = (ssim_map * valid_mask).sum() / (num_valid + 1e-8)
            ssim_value = torch.clamp(ssim_value, 0.0, 1.0)
            loss_ms_ssim = loss_ms_ssim + weight * (1.0 - ssim_value)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            p = pred_u.float()
            g = gt_u.float()
            m = valid_mask.float()
            denom = torch.clamp(g, min=self.mape_denom_eps)
            mape_map = torch.abs(p - g) / denom
            mape_map = torch.clamp(mape_map, max=self.mape_cap)
            loss_mape = (mape_map * m).sum() / (m.sum() + 1e-8)
        loss_extremes = pred_u.new_tensor(0.0)
        loss_peak_under = pred_u.new_tensor(0.0)
        loss_valley_over = pred_u.new_tensor(0.0)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            p32 = pred_u.float()
            g32 = gt_u.float()
            m32 = valid_mask.float()
            peak_map = peak_cell_mask_map.to(p32.dtype)
            if (peak_map * m32).sum() > 0:
                under = F.relu(g32 - p32)  # GT > Pred
                peak_w = (1.0 + self.peak_w_scale * g32).detach()
                denom_peak = (peak_map * m32).sum().clamp_min(1e-6)
                loss_peak_under = (under * peak_w * peak_map * m32).sum() / denom_peak
            valley_map = valley_cell_mask_map.to(p32.dtype)
            if (valley_map * m32).sum() > 0:
                over = F.relu(p32 - g32)  # Pred > GT
                denom_valley = (valley_map * m32).sum().clamp_min(1e-6)
                loss_valley_over = (over * valley_map * m32).sum() / denom_valley
            loss_extremes = loss_peak_under + loss_valley_over
            loss_extremes = torch.nan_to_num(loss_extremes, nan=0.0, posinf=1e3, neginf=0.0)
        loss_grid_hist = pred_u.new_tensor(0.0)
        loss_grad = pred_u.new_tensor(0.0)
        loss_grid_contrast = pred_u.new_tensor(0.0)
        if boxes_list is not None:
            device = pred.device
            rois = []
            for b_idx, boxes in enumerate(boxes_list):
                if boxes is None or len(boxes) == 0:
                    continue
                boxes = boxes.to(device)
                idx_col = torch.full((len(boxes), 1), b_idx, device=device, dtype=boxes.dtype)
                rois.append(torch.cat([idx_col, boxes], dim=1))
            if len(rois) > 0:
                rois_cat = torch.cat(rois, dim=0)
                pred_roi = roi_align(pred * valid_mask, rois_cat, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1)
                gt_roi = roi_align(gt * valid_mask, rois_cat, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1)
                m_roi = roi_align(valid_mask, rois_cat, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=4).view(-1).clamp_min(1e-6)
                pred_mean = pred_roi / m_roi
                gt_mean = gt_roi / m_roi
                loss_grad = self.grad_loss(pred_u, gt_u, valid_mask, boxes_list)
                hp = self._soft_hist(pred_mean, bins=self.hist_bins, sigma=self.hist_sigma)
                hg = self._soft_hist(gt_mean, bins=self.hist_bins, sigma=self.hist_sigma)
                loss_grid_hist = F.l1_loss(hp, hg)
                pairs_i, pairs_j = [], []
                cursor = 0
                for boxes in boxes_list:
                    k = 0 if (boxes is None) else len(boxes)
                    if k == 0:
                        continue
                    bcpu = boxes.detach().round().long().cpu()
                    xs0 = bcpu[:, 0].tolist()
                    ys0 = bcpu[:, 1].tolist()
                    xs1 = bcpu[:, 2].tolist()
                    ys1 = bcpu[:, 3].tolist()
                    loc = {(xs0[t], ys0[t]): (cursor + t) for t in range(k)}
                    for t in range(k):
                        rk = (xs1[t], ys0[t])
                        if rk in loc:
                            pairs_i.append(cursor + t)
                            pairs_j.append(loc[rk])
                        dk = (xs0[t], ys1[t])
                        if dk in loc:
                            pairs_i.append(cursor + t)
                            pairs_j.append(loc[dk])
                    cursor += k
                if len(pairs_i) > 0:
                    pi = torch.tensor(pairs_i, device=device, dtype=torch.long)
                    pj = torch.tensor(pairs_j, device=device, dtype=torch.long)
                    pred_diff = (pred_mean[pi] - pred_mean[pj]).abs()
                    gt_diff = (gt_mean[pi] - gt_mean[pj]).abs()
                    w = (1.0 + 50 * gt_diff.clamp(0.0, 1.0)).detach()
                    loss_grid_contrast = ((pred_diff - gt_diff).abs() * w).mean()
                else:
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        p = pred_u.float()
                        g = gt_u.float()
                        m = valid_mask.float()
                        total_c = p.new_tensor(0.0)
                        cnt = 0
                        for size in [(24, 24), (48, 48)]:
                            p_num = F.adaptive_avg_pool2d(p * m, size)
                            g_num = F.adaptive_avg_pool2d(g * m, size)
                            m_s = F.adaptive_avg_pool2d(m, size).clamp_min(1e-6)
                            p_s = p_num / m_s
                            g_s = g_num / m_s
                            dp_x = p_s[:, :, :, 1:] - p_s[:, :, :, :-1]
                            dg_x = g_s[:, :, :, 1:] - g_s[:, :, :, :-1]
                            mx = m_s[:, :, :, 1:] * m_s[:, :, :, :-1]
                            wx = (1.0 + 100.0 * dg_x.abs().clamp(0.0, 1.0)).clamp(max=50.0).detach()
                            loss_x = ((dp_x - dg_x).abs() * wx * mx).sum() / (mx.sum() + 1e-6)
                            dp_y = p_s[:, :, 1:, :] - p_s[:, :, :-1, :]
                            dg_y = g_s[:, :, 1:, :] - g_s[:, :, :-1, :]
                            my = m_s[:, :, 1:, :] * m_s[:, :, :-1, :]
                            wy = (1.0 + 100.0 * dg_y.abs().clamp(0.0, 1.0)).clamp(max=50.0).detach()
                            loss_y = ((dp_y - dg_y).abs() * wy * my).sum() / (my.sum() + 1e-6)
                            total_c = total_c + (loss_x + loss_y)
                            cnt += 1
                        loss_grid_contrast = total_c / max(1, cnt)
        loss_logmse = torch.log(loss_mse.clamp(min=1e-9) + 1e-8) - math.log(1e-8)
        loss_logmse = torch.nan_to_num(loss_logmse, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_grad_supp = pred_u.new_tensor(0.0)
        if input_img is not None:
            loss_grad_supp = self._compute_gradient_suppress_loss(
                pred_u, gt_u, input_img, valid_mask, k_percent=self.gradient_suppress_k
            )
        parts = {
            "loss_l1": loss_l1,
            "loss_mape": loss_mape,
            "loss_mse": loss_mse,
            "loss_ms_ssim": loss_ms_ssim,
            "loss_grid_hist": loss_grid_hist,
            "loss_grad": loss_grad,
            "loss_grid_contrast": loss_grid_contrast,
            "loss_extremes": loss_extremes,
            "loss_logmse": loss_logmse,
        }
        bad = [k for k, v in parts.items() if (v is not None) and (not torch.isfinite(v).all().item())]
        m_soft = F.avg_pool2d(valid_mask, kernel_size=3, stride=1, padding=1)
        pf = pred_u * m_soft
        gf = gt_u * m_soft
        pf = F.interpolate(pf, size=(self.freq_size, self.freq_size), mode='bilinear', align_corners=False).float()
        gf = F.interpolate(gf, size=(self.freq_size, self.freq_size), mode='bilinear', align_corners=False).float()
        Fp = torch.fft.rfft2(pf, norm='ortho')
        Fg = torch.fft.rfft2(gf, norm='ortho')
        Ap = torch.pow(torch.abs(Fp) + 1e-8, 0.5)
        Ag = torch.pow(torch.abs(Fg) + 1e-8, 0.5)
        loss_freq = (torch.abs(Ap - Ag) * self.freq_weight).mean()
        if bad:
            print(f"\n[NaN/Inf][BlockAwareLoss] bad_parts={bad}")
            for k in bad:
                v = parts[k].detach().float()
                if v.numel() == 1:
                    print(f"  - {k}: {float(v.item())}")
                else:
                    print(f"  - {k}: min={float(v.min().item())} max={float(v.max().item())}")
            print(f"  - pred_u: min={float(pred_u.float().min().item())} max={float(pred_u.float().max().item())}")
            print(f"  - gt_u:   min={float(gt_u.float().min().item())} max={float(gt_u.float().max().item())}")
            print(f"  - num_valid={float(num_valid.item())}")
        loss_l1 = torch.nan_to_num(loss_l1, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_mape = torch.nan_to_num(loss_mape, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_mse = torch.nan_to_num(loss_mse, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_ms_ssim = torch.nan_to_num(loss_ms_ssim, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_grid_hist = torch.nan_to_num(loss_grid_hist, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_grad = torch.nan_to_num(loss_grad, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_grid_contrast = torch.nan_to_num(loss_grid_contrast, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_extremes = torch.nan_to_num(loss_extremes, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_freq = torch.nan_to_num(loss_freq, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        loss_grad_supp = torch.nan_to_num(loss_grad_supp, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        total = (
                w_l1 * loss_l1 +
                w_mape * loss_mape +
                w_mse * loss_mse +
                w_ms_ssim * loss_ms_ssim +
                w_grid_hist * loss_grid_hist +
                w_grad * loss_grad +
                w_grid_contrast * loss_grid_contrast +
                w_extremes * loss_extremes +
                w_freq * loss_freq +
                w_grad_supp * loss_grad_supp
        )
        if self.training:
            # 记录处理后的值到 stats 以便打印
            self.loss_stats['loss_l1'].append(loss_l1.item())
            self.loss_stats['loss_mape'].append(loss_mape.item())
            self.loss_stats['loss_mse'].append(loss_mse.item())
            self.loss_stats['loss_ms_ssim'].append(loss_ms_ssim.item())
            self.loss_stats['loss_grid_hist'].append(loss_grid_hist.item())
            self.loss_stats['loss_grad'].append(loss_grad.item())
            self.loss_stats['loss_grid_contrast'].append(loss_grid_contrast.item())
            self.loss_stats['loss_extremes'].append(loss_extremes.item())
            self.loss_stats['loss_freq'].append(loss_freq.item())
            self.loss_stats['loss_grad_supp'].append(loss_grad_supp.item())
            self.loss_stats['total'].append(total.item())
        return total
    def print_epoch_stats(self, epoch):
        if not self.loss_stats['total']:
            print("No loss data collected in this epoch.")
            return
        import numpy as np
        loss_components = [
            ("loss_l1", w_l1),
            ("loss_mape", w_mape),
            ("loss_mse", w_mse),
            ("loss_ms_ssim", w_ms_ssim),
            ("loss_grid_hist", w_grid_hist),
            ("loss_grad", w_grad),
            ("loss_grid_contrast", w_grid_contrast),
            ("loss_extremes", w_extremes),
            ("loss_freq", w_freq),
            ("loss_grad_supp", w_grad_supp),
        ]
        stats_cache = {}
        total_weighted_sum_from_means = 0.0
        for name, weight in loss_components:
            if name in self.loss_stats and self.loss_stats[name]:
                values = np.array(self.loss_stats[name])
                # 过滤掉nan和inf
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    mean_val = np.mean(values)
                    weighted_val = mean_val * weight
                    total_weighted_sum_from_means += weighted_val
                    stats_cache[name] = {
                        "mean": mean_val,
                        "weighted": weighted_val,
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "valid": True
                    }
                else:
                    stats_cache[name] = {"valid": False}
            else:
                stats_cache[name] = {"valid": False}
        header_str = (
            f"{'Loss Component':<20} {'Weight':>8} | "
            f"{'Mean':>12} {'Weighted':>12} {'%Total':>8} | "
            f"{'Std':>12} {'Min':>12} {'Max':>12}"
        )
        print(f"{'=' * len(header_str)}")
        print(f"EPOCH {epoch} - Loss Statistics Summary")
        print(f"{'=' * len(header_str)}")
        print(header_str)
        print(f"{'-' * len(header_str)}")
        for name, weight in loss_components:
            info = stats_cache[name]
            if info["valid"]:
                # 计算占比
                pct = 0.0
                if total_weighted_sum_from_means > 1e-9:
                    pct = (info["weighted"] / total_weighted_sum_from_means) * 100
                print(
                    f"{name:<20} {weight:>8.1f} | "
                    f"{info['mean']:>12.6f} {info['weighted']:>12.6f} {pct:>7.1f}% | "
                    f"{info['std']:>12.6f} {info['min']:>12.6f} {info['max']:>12.6f}"
                )
            else:
                # 无效数据或权重为0且无数据
                print(
                    f"{name:<20} {weight:>8.1f} | {'NaN/Inf':>12} {'0.0000':>12} {'0.0%':>8} | {'-':>12} {'-':>12} {'-':>12}")
        print(f"{'=' * len(header_str)}")
        if self.loss_stats['total']:
            total_values = np.array(self.loss_stats['total'])
            total_values = total_values[np.isfinite(total_values)]
            if len(total_values) > 0:
                mean_t = np.mean(total_values)
                print(
                    f"{'TOTAL (Tracked)':<20} {'-':>8} | "
                    f"{mean_t:>12.6f} {mean_t:>12.6f} {'100.0%':>8} | "
                    f"{np.std(total_values):>12.6f} {np.min(total_values):>12.6f} {np.max(total_values):>12.6f}"
                )
        print(f"{'=' * len(header_str)}")
        print(f"Total batches processed: {len(self.loss_stats['total'])}")
        nan_counts = {}
        for name in self.loss_stats:
            if self.loss_stats[name]:
                values = np.array(self.loss_stats[name])
                nan_count = np.sum(~np.isfinite(values))
                if nan_count > 0:
                    nan_counts[name] = nan_count
        if nan_counts:
            print(f"\n⚠️  Warning: NaN/Inf detected in following components:")
            for name, count in nan_counts.items():
                print(f"   - {name}: {count} batches")
    def reset_stats(self):
        """清空统计数据"""
        for key in self.loss_stats:
            self.loss_stats[key] = []
def visualize_strict(model, sample, epoch, save_dir, device):
    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    inp = sample['input'].to(device).unsqueeze(0)
    he = sample['he'].to(device).unsqueeze(0)
    gt = sample['gt'].to(device).unsqueeze(0)
    boxes = sample['boxes'].to(device)
    h_lines = [sample['h_lines']]
    v_lines = [sample['v_lines']]
    with torch.inference_mode():
        raw_pred, _ = model(inp, he, h_lines, v_lines, [boxes])
    def to_np(t):
        arr = t.squeeze().detach().cpu().numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(arr, 0.0, 1.0)
    img_gt = to_np(gt)
    img_pred = to_np(raw_pred)
    if np.isnan(img_pred).any() or np.isinf(img_pred).any():
        img_pred = np.nan_to_num(img_pred, nan=0.0, posinf=1.0, neginf=0.0)
    fig = None
    abs_error = np.abs(img_pred - img_gt)
    signed_error = img_pred - img_gt
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        im0 = axes[0, 0].imshow(img_gt, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[0, 0].set_title("Ground Truth (Block)")
        fig.colorbar(im0, ax=axes[0, 0])
        im1 = axes[0, 1].imshow(img_pred, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[0, 1].set_title(f"Prediction (Ep {epoch})")
        fig.colorbar(im1, ax=axes[0, 1])
        im2 = axes[0, 2].imshow(abs_error, cmap='inferno', vmin=0,
                                vmax=np.nanmax(abs_error) if np.nanmax(abs_error) > 0 else 1.0)
        axes[0, 2].set_title("Absolute Error Map")
        fig.colorbar(im2, ax=axes[0, 2])
        limit = max(np.abs(signed_error.min()), np.abs(signed_error.max()), 0.1)
        im3 = axes[1, 0].imshow(signed_error, cmap='seismic', vmin=-limit, vmax=limit)
        axes[1, 0].set_title("Signed Error Map")
        fig.colorbar(im3, ax=axes[1, 0])
        he_np = to_np(sample['he'])
        if he_np.ndim == 3: he_np = he_np.transpose(1, 2, 0)
        axes[1, 1].imshow(he_np)
        axes[1, 1].set_title("HE Reference")
        axes[1, 2].imshow(to_np(sample['input']), cmap='gray')
        axes[1, 2].set_title("Input Sampling")
        for ax in axes.flatten(): ax.axis('off')
        target_file = save_path / f"vis_epoch_{epoch:03d}.png"
        fig.savefig(str(target_file), dpi=150, bbox_inches='tight', pad_inches=0.1, facecolor='white',
                    transparent=False)
    except Exception as e:
        print(f"[Error] Failed to save visualization for epoch {epoch}: {e}")
    finally:
        if fig is not None:
            plt.close(fig)
def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    inputs = torch.stack([item['input'] for item in batch])
    hes = torch.stack([item['he'] for item in batch])
    gts = torch.stack([item['gt'] for item in batch])
    boxes_list = [item['boxes'] for item in batch]
    return {
        'input': inputs, 'he': hes, 'gt': gts,
        'boxes': boxes_list,
        'h_lines': [b['h_lines'] for b in batch],
        'v_lines': [b['v_lines'] for b in batch],
        'names': [b['name'] for b in batch]
    }
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss_value):
        loss_value = float(loss_value)
        if self.best_loss is None:
            self.best_loss = loss_value
            self.counter = 0
            return
        if (self.best_loss - loss_value) > self.min_delta:
            self.best_loss = loss_value
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience} (best_loss={self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
def masked_psnr(pred, gt, valid_mask, eps=1e-8):
    """
    pred, gt, valid_mask: [B, 1, H, W]，值域在[0,1]
    PSNR = 10*log10(MAX^2 / MSE), 这里 MAX=1
    只在 valid_mask=1 的像素上计算 MSE
    """
    mse = ((pred - gt) ** 2 * valid_mask).sum() / (valid_mask.sum() + eps)
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return torch.nan_to_num(psnr, nan=0.0, posinf=100.0, neginf=0.0)
def train_pipeline():
    save_dir = Path("/mnt/xhdisk/02 MSI/SRP/【监控】低分辨")
    save_dir.mkdir(parents=True, exist_ok=True)
    linux_root_dir = r"/mnt/xhdisk/02 MSI/SRP/SR"
    csv_file_path = r"/mnt/xhdisk/02 MSI/SRP/SR/train_data.csv"
    TOTAL_EPOCHS = 100
    WARMUP_EPOCHS = 3
    ACCUM_STEPS = 8
    BATCH_SIZE = 1
    ACCUM_START_EPOCH = 8
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    full_ds = MSIDataset(csv_file_path, root_dir=linux_root_dir, cache_data=True)
    id_to_indices = {}
    for idx, s in enumerate(full_ds.samples):
        k = s['name']
        if k not in id_to_indices:
            id_to_indices[k] = []
        id_to_indices[k].append(idx)
    all_ids = list(id_to_indices.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(all_ids)
    n_train_ids = int(alpha_train * len(all_ids))
    n_train_ids = max(1, min(n_train_ids, len(all_ids) - 1))
    train_ids = all_ids[:n_train_ids]
    val_ids = all_ids[n_train_ids:]
    train_indices = []
    for k in train_ids:
        train_indices.extend(id_to_indices[k])
    val_indices = []
    for k in val_ids:
        val_indices.extend(id_to_indices[k])
    train_dataset = torch.utils.data.Subset(full_ds, train_indices)
    val_dataset = torch.utils.data.Subset(full_ds, val_indices)
    train_dl = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, collate_fn=custom_collate_fn,
        pin_memory=True, prefetch_factor=1, persistent_workers=True
    )
    val_dl = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, collate_fn=custom_collate_fn,
        pin_memory=True, prefetch_factor=1, persistent_workers=True
    )
    generator = CascadeInpaintingNet().to(device)
    generator = generator.to(memory_format=torch.channels_last)
    discriminator = PatchDiscriminator(in_channels=1).to(device)
    print("-" * 60)
    print("Model Architecture Summary:")
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Total Parameters:  {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable Ratio:   {trainable_params / total_params:.2%}")
    print("-" * 60)
    l1_loss_fn = BlockAwareLoss().to(device)
    gan_loss_fn = nn.MSELoss().to(device)
    optimizer_G = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=5e-4, weight_decay=1e-4, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=2e-4, weight_decay=1e-4, betas=(0.5, 0.999)
    )
    scaler = torch.amp.GradScaler('cuda', init_scale=2. ** 10, growth_interval=2000)

    def safe_scaler_update():
        try:
            scaler.update()
        except AssertionError:
            scaler.update(scaler.get_scale())

    g_optimizer_has_stepped = False
    early_stopping = EarlyStopping(patience=1000, min_delta=1e-4)

    def g_steps_in_epoch(e):
        a = 1 if e < ACCUM_START_EPOCH else ACCUM_STEPS
        return math.ceil(len(train_dl) / a)

    total_steps = sum(g_steps_in_epoch(e) for e in range(1, TOTAL_EPOCHS + 1))
    warmup_steps = sum(g_steps_in_epoch(e) for e in range(1, WARMUP_EPOCHS + 1))

    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler_G = get_cosine_schedule_with_warmup(
        optimizer_G,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    best_train_psnr = -1e9
    best_val_psnr = -1e9

    def check_model_params(model):
        """检查模型参数是否包含 NaN/Inf"""
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                return False, name
        return True, None

    def try_load_rollback_checkpoint(model, save_dir, device):
        """尝试加载最近的权重以恢复训练"""
        # 优先加载 best_train，其次 best_val
        p1 = save_dir / "model_best_train_psnr.pth"
        p2 = save_dir / "model_best_val_psnr.pth"
        target = p1 if p1.exists() else (p2 if p2.exists() else None)
        if target:
            print(f"[ROLLBACK] Loading checkpoint from {target} to fix NaN model...")
            try:
                state = torch.load(target, map_location=device)
                model.load_state_dict(state)
                return True
            except Exception as e:
                print(f"[ROLLBACK ERROR] {e}")
                return False
        return False
    for epoch in range(1, TOTAL_EPOCHS + 1):
        torch.cuda.empty_cache()
        generator.train()
        discriminator.train()
        epoch_error_count = 0
        max_errors_per_epoch = 5
        train_psnr_accum = torch.tensor(0.0, device=device)
        train_items_count = 0
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        valid_steps = 0
        accum_steps_this_epoch = 1 if epoch < ACCUM_START_EPOCH else ACCUM_STEPS
        optimizer_G.zero_grad(set_to_none=True)
        accum_counter = 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch}")
        for batch in pbar:
            if batch is None:
                continue
            inp = batch['input'].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            he = batch['he'].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            gt = batch['gt'].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            h_lines = batch['h_lines']
            v_lines = batch['v_lines']
            boxes_list = [b.to(device) for b in batch['boxes']]
            if not (torch.isfinite(inp).all() and torch.isfinite(he).all() and torch.isfinite(gt).all()):
                print(f"\n[WARNING] Skipping batch due to NaN/Inf in input data")
                torch.cuda.empty_cache()
                continue
            for p in discriminator.parameters():
                p.requires_grad = False
            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    final_pred_blocky, coarse_pred = generator(inp, he, h_lines, v_lines, boxes_list)
                    if not torch.isfinite(final_pred_blocky).all():
                        raise ValueError("NaN/Inf in generator output")
                    loss_content_final = l1_loss_fn(final_pred_blocky, gt, boxes_list=boxes_list, input_img=inp)
                    coarse_pred_blocky = apply_grid_median_smoothing(coarse_pred, h_lines, v_lines)
                    loss_content_coarse = l1_loss_fn(coarse_pred_blocky, gt, boxes_list=boxes_list, input_img=inp)
                    loss_content = loss_content_final + 0.5 * loss_content_coarse
                    pred_fake = discriminator(final_pred_blocky)
                    target_real = torch.ones_like(pred_fake)
                    loss_gan_g = gan_loss_fn(pred_fake, target_real)
                    loss_g_raw = loss_content + 0.5 * loss_gan_g
                    loss_g = loss_g_raw / accum_steps_this_epoch
                    if not torch.isfinite(loss_g_raw).all():
                        raise ValueError("NaN/Inf in loss")
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        valid_mask_train = (gt.float() > 1e-6).float()
                        psnr_train = masked_psnr(final_pred_blocky.float(), gt.float(), valid_mask_train)
                    bs_curr = int(final_pred_blocky.shape[0])
                    train_psnr_accum += psnr_train.detach() * bs_curr
                    train_items_count += bs_curr
            except (ValueError, RuntimeError) as e:
                epoch_error_count += 1
                print(f"\n[ERROR] Batch failed (#{epoch_error_count}): {e}")
                optimizer_G.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if epoch_error_count >= max_errors_per_epoch:
                    print(f"\n[🚨 CRITICAL] Epoch {epoch} encountered {epoch_error_count} errors. Jumping out of epoch...")
                    if try_load_rollback_checkpoint(generator, save_dir, device):
                        print("[ACTION] Rollback successful. Resetting Optimizer momentum and Scaler.")
                        optimizer_G = torch.optim.AdamW(
                            filter(lambda p: p.requires_grad, generator.parameters()),
                            lr=optimizer_G.param_groups[0]['lr'],
                            weight_decay=1e-3, betas=(0.5, 0.999)
                        )
                        scheduler_G = get_cosine_schedule_with_warmup(
                            optimizer_G, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=0.5
                        )
                        scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
                    break
                current_scale = scaler.get_scale()
                new_scale = max(1.0, current_scale * 0.25)
                print(f"[ACTION] Reducing Scale: {current_scale} -> {new_scale}")
                scaler.update(new_scale)
                continue
            loss_pack_g = {
                "final_pred_blocky": final_pred_blocky,
                "coarse_pred": coarse_pred,
                "loss_content_final": loss_content_final,
                "loss_content_coarse": loss_content_coarse,
                "loss_content": loss_content,
                "loss_gan_g": loss_gan_g,
                "loss_g_raw": loss_g_raw,
            }
            bad_g = {k: v for k, v in loss_pack_g.items()
                     if (v is not None) and (not torch.isfinite(v).all().item())}
            if bad_g:
                print(f"\n[NaN/Inf][G][epoch={epoch}] bad={list(bad_g.keys())}")
                optimizer_G.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                safe_scaler_update()
                continue
            scaler.scale(loss_g).backward()
            accum_counter += 1
            did_g_step = False
            g_optimizer_stepped = False
            if (accum_counter % accum_steps_this_epoch) == 0:
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                has_inf_nan = False
                for p in generator.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            has_inf_nan = True
                            break
                if has_inf_nan:
                    print(f"[WARNING] Detected Inf/NaN in gradients, skipping G update")
                    optimizer_G.zero_grad(set_to_none=True)
                    scaler.update()
                    torch.cuda.empty_cache()
                else:
                    scaler.step(optimizer_G)
                    g_optimizer_stepped = True
                    optimizer_G.zero_grad(set_to_none=True)
                    scaler.update()
                did_g_step = True
            for p in discriminator.parameters():
                p.requires_grad = True
            optimizer_D.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                fake_img = final_pred_blocky.detach()
                real_img = gt
                pred_fake_d = discriminator(fake_img)
                target_fake = torch.zeros_like(pred_fake_d)
                loss_d_fake = gan_loss_fn(pred_fake_d, target_fake)
                pred_real_d = discriminator(real_img)
                target_real = torch.ones_like(pred_real_d)
                loss_d_real = gan_loss_fn(pred_real_d, target_real)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
            scaler.scale(loss_d).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler.step(optimizer_D)
            scaler.update()
            if did_g_step and g_optimizer_stepped:
                g_optimizer_has_stepped = True
                scheduler_G.step()
            epoch_loss_g += float(loss_g_raw.item())
            epoch_loss_d += float(loss_d.item())
            valid_steps += 1
            pbar.set_postfix({'G_Loss': f"{loss_g_raw.item():.4f}", 'D_Loss': f"{loss_d.item():.4f}"})
        if epoch_error_count >= max_errors_per_epoch:
            l1_loss_fn.reset_stats()
            print(f"--- Skip current corrupt Epoch {epoch} and jump to next ---")
            continue
        rem = (accum_counter % accum_steps_this_epoch)
        if rem != 0:
            scaler.unscale_(optimizer_G)
            if accum_steps_this_epoch > 1:
                scale_fix = float(accum_steps_this_epoch) / float(rem)
                for p in generator.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale_fix)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            has_inf_nan = False
            for p in generator.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        has_inf_nan = True
                        break
            if not has_inf_nan:
                scaler.step(optimizer_G)
                g_optimizer_has_stepped = True
                scheduler_G.step()
            optimizer_G.zero_grad(set_to_none=True)
            scaler.update()
        if (not math.isfinite(epoch_loss_g)) or (not math.isfinite(epoch_loss_d)) or (valid_steps == 0):
            print(f"Skipping epoch {epoch} due to NaN/Inf (or no valid steps).")
            continue
        avg_train_loss_g = epoch_loss_g / valid_steps
        avg_train_loss_d = epoch_loss_d / valid_steps
        if train_items_count > 0:
            avg_train_psnr = (train_psnr_accum / train_items_count).item()
        else:
            avg_train_psnr = 0.0
        if avg_train_psnr > best_train_psnr:
            best_train_psnr = avg_train_psnr
            print(f"[⭐⭐⭐⭐⭐] New Best TRAIN PSNR: {best_train_psnr:.5f} -> Saving model_best_train_psnr.pth")
            torch.save(generator.state_dict(), save_dir / "model_best_train_psnr.pth")
        print(f"[Epoch {epoch}/{TOTAL_EPOCHS}] G Loss: {avg_train_loss_g:.6f} | "
              f"D Loss: {avg_train_loss_d:.6f} | LR: {optimizer_G.param_groups[0]['lr']:.2e}")
        if epoch % 1 == 0:
            vis_idx = np.random.randint(len(val_dataset))
            visualize_strict(generator, val_dataset[vis_idx], epoch, str(save_dir), device)
        generator.eval()
        torch.cuda.empty_cache()
        val_mae_accum = torch.tensor(0.0, device=device)
        val_psnr_accum = torch.tensor(0.0, device=device)
        total_items_count = 0
        with torch.inference_mode():
            for batch_idx, batch in enumerate(val_dl):
                if batch_idx >= val_break_preview:
                    break
                if batch is None:
                    continue
                inp = batch['input'].to(device, non_blocking=True)
                he = batch['he'].to(device, non_blocking=True)
                gt = batch['gt'].to(device, non_blocking=True)
                boxes_list = [b.to(device) for b in batch['boxes']]
                pred, _ = generator(inp, he, batch['h_lines'], batch['v_lines'], boxes_list)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
                pred = torch.clamp(pred, 0.0, 1.0)
                batch_size_curr = pred.shape[0]
                valid_mask = (gt > 1e-6).float()
                num_valid = valid_mask.sum() + 1e-6
                pred_masked = pred * valid_mask
                gt_masked = gt * valid_mask
                mae_raw = torch.abs(pred_masked - gt_masked) * valid_mask
                mae_val = torch.nan_to_num(mae_raw, nan=0.0).sum() / num_valid
                mae_val = torch.clamp(mae_val, 0.0, 1.0)
                psnr_val = masked_psnr(pred, gt, valid_mask)
                val_psnr_accum += psnr_val * batch_size_curr
                val_mae_accum += mae_val * batch_size_curr
                total_items_count += batch_size_curr
        if total_items_count > 0:
            avg_psnr = (val_psnr_accum / total_items_count).item()
            avg_mae = (val_mae_accum / total_items_count).item()
        else:
            avg_psnr = 0.0
            avg_mae = 1.0
        print(f"Ep {epoch} Val -> PSNR: {avg_psnr:.5f} | MAE: {avg_mae:.5f}")
        l1_loss_fn.print_epoch_stats(epoch)
        l1_loss_fn.reset_stats()
        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            print(f"[🟢🟢🟢🟢🟢] New Best VAL PSNR: {best_val_psnr:.5f} -> Saving model_best_val_psnr.pth")
            torch.save(generator.state_dict(), save_dir / "model_best_val_psnr.pth")
        early_stopping(avg_train_loss_g)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        generator.train()
        discriminator.train()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train_pipeline()