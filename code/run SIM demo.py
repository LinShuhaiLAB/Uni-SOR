import os
import sys
from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
############################################################
#                                                          #
#             You only need to modify here                 #
#                                                          #
############################################################
INPUT_PATH = str(Path(__file__).resolve().parent.parent / "demo for SIM deblurring" / "input.jpg")
WEIGHT_PATH = str(Path(__file__).resolve().parent.parent / "weights of SRP" / "ER.pth")











GPU = "0"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
class AdaptiveFrequencyNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(1.0))
        self.eps = 1e-8
    def forward(self, fft_features):
        energy = torch.sum(fft_features ** 2, dim=(1, 2, 3), keepdim=True)
        energy_norm = fft_features / torch.sqrt(energy + self.eps)
        log_compressed = torch.log1p(torch.abs(energy_norm) * torch.exp(self.log_scale))
        return log_compressed / (log_compressed.max() + self.eps)
class CompleteZernikePolynomials(nn.Module):
    def __init__(self, size=256, max_order=6):
        super().__init__()
        self.size = size
        self.max_order = max_order
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing="ij",
        )
        self.register_buffer("rho", torch.sqrt(x ** 2 + y ** 2))
        self.register_buffer("theta", torch.atan2(y, x))
        self.register_buffer("mask", (self.rho <= 1.0).float())
        self._precompute_radial_powers()
        self.num_terms = self._calculate_num_terms(max_order)
    def _calculate_num_terms(self, max_order):
        return (max_order + 1) * (max_order + 2) // 2
    def _precompute_radial_powers(self):
        powers = [self.rho ** i for i in range(9)]
        self.register_buffer("rho_powers", torch.stack(powers, dim=0))
    def radial_polynomial_optimized(self, n, m):
        if (n - m) % 2 != 0:
            return torch.zeros_like(self.rho)
        r = torch.zeros_like(self.rho)
        for k in range((n - m) // 2 + 1):
            coef = (-1) ** k * self._factorial(n - k) / (
                self._factorial(k)
                * self._factorial((n + m) // 2 - k)
                * self._factorial((n - m) // 2 - k)
            )
            r += coef * self.rho_powers[n - 2 * k]
        return r
    def _factorial(self, n):
        if n <= 1:
            return 1.0
        return float(np.exp(sum(np.log(np.arange(1, n + 1)))))
    def _noll_to_nm(self, j):
        noll_map = {
            1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2),
            6: (2, 2), 7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3),
            11: (4, 0), 12: (4, 2), 13: (4, -2), 14: (4, 4), 15: (4, -4),
            16: (5, 1), 17: (5, -1), 18: (5, 3), 19: (5, -3), 20: (5, 5),
            21: (5, -5), 22: (6, 0), 23: (6, -2), 24: (6, 2), 25: (6, -4),
            26: (6, 4), 27: (6, -6), 28: (6, 6), 29: (7, -1), 30: (7, 1),
            31: (7, -3), 32: (7, 3), 33: (7, -5), 34: (7, 5), 35: (7, -7),
            36: (7, 7), 37: (8, 0),
        }
        return noll_map.get(j, (0, 0))
    def forward(self, coefficients):
        b, n_coeffs = coefficients.shape
        n_coeffs = min(n_coeffs, self.num_terms)
        wavefront = torch.zeros(
            b, self.size, self.size,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        for j in range(1, n_coeffs + 1):
            n, m = self._noll_to_nm(j)
            r_nm = self.radial_polynomial_optimized(n, abs(m))
            if m > 0:
                z = r_nm * torch.cos(m * self.theta)
            elif m < 0:
                z = r_nm * torch.sin(abs(m) * self.theta)
            else:
                z = r_nm
            if m != 0:
                norm = torch.sqrt(torch.tensor(2.0 * (n + 1), device=self.rho.device, dtype=self.rho.dtype))
            else:
                norm = torch.sqrt(torch.tensor(float(n + 1), device=self.rho.device, dtype=self.rho.dtype))
            wavefront += coefficients[:, j - 1].view(b, 1, 1) * z * norm * self.mask
        return wavefront
class LightweightEstimator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.freq_normalizer = AdaptiveFrequencyNormalizer()
        def dw_conv(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
            )
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels + 2, 32, 3, 1, 1), nn.ReLU(inplace=True), dw_conv(32, 32))
        self.enc2 = nn.Sequential(dw_conv(32, 64, stride=2), dw_conv(64, 64))
        self.enc3 = nn.Sequential(dw_conv(64, 128, stride=2), dw_conv(128, 128))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True))
        self.depth_head = nn.Conv2d(32, 1, 3, 1, 1)
        self.conf_head = nn.Conv2d(32, 1, 3, 1, 1)
        self.sigma_head = nn.Conv2d(32, 1, 3, 1, 1)
    def extract_frequency_features(self, x):
        b, c, h, w = x.shape
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        fft_mag = torch.abs(torch.fft.rfft2(gray, dim=(-2, -1)))
        fft_normalized = self.freq_normalizer(fft_mag)
        fft_spatial = F.interpolate(fft_normalized, size=(h, w), mode="bilinear", align_corners=False)
        high_freq = fft_spatial - F.avg_pool2d(fft_spatial, kernel_size=5, stride=1, padding=2)
        low_freq = F.adaptive_avg_pool2d(fft_spatial, (1, 1)).expand(-1, 1, h, w)
        return torch.cat([low_freq, high_freq], dim=1)
    def forward(self, x):
        x_combined = torch.cat([x, self.extract_frequency_features(x)], dim=1)
        f1 = self.enc1(x_combined)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        d2 = self.dec2(f3)
        d1 = self.dec1(d2 + f2)
        depth = torch.sigmoid(self.depth_head(d1 + f1))
        conf = torch.sigmoid(self.conf_head(d1 + f1))
        sigma = torch.sigmoid(self.sigma_head(d1 + f1))
        return depth, conf, sigma, {"f1": f1, "f2": f2, "f3": f3}
class HighResReconstructor(nn.Module):
    def __init__(self, in_channels=3, base_c=32):
        super().__init__()
        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels + 3, base_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_c, base_c, 3, 1, 1),
            nn.ReLU(),
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(base_c + 64, base_c * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_c * 2, base_c * 2, 3, 1, 1),
            nn.ReLU(),
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(base_c * 2 + 128, base_c * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_c * 4, base_c * 4, 3, 1, 1),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(base_c * 4, base_c * 2, 4, 2, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base_c * 2, base_c, 4, 2, 1), nn.ReLU())
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_c, base_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_c, in_channels, 3, 1, 1),
        )
    def forward(self, x, d, c, s, feats):
        weighted_d = d * c
        x_with_maps = torch.cat([x, weighted_d, c, s], dim=1)
        f1_fused = self.fusion1(x_with_maps)
        f1_down = F.avg_pool2d(f1_fused, kernel_size=2, stride=2)
        f2_fused = self.fusion2(torch.cat([f1_down, feats["f2"]], dim=1))
        f2_down = F.avg_pool2d(f2_fused, kernel_size=2, stride=2)
        f3_fused = self.fusion3(torch.cat([f2_down, feats["f3"]], dim=1))
        d3 = self.dec3(f3_fused) + f2_fused
        d2 = self.dec2(d3) + f1_fused
        return self.out_conv(d2)
class ParameterRefiner(nn.Module):
    def __init__(self, in_channels=3 + 3 + 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )
        nn.init.constant_(self.net[-2].weight, 0)
        nn.init.constant_(self.net[-2].bias, 0)
    def forward(self, x, coarse_sharp, coarse_maps):
        inp = torch.cat(
            [
                x,
                coarse_sharp.detach(),
                coarse_maps["depth"],
                coarse_maps["confidence"],
                coarse_maps["sigma"],
            ],
            dim=1,
        )
        return self.net(inp) * 0.2
class MultiScalePhysicsConsistency(nn.Module):
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
    def downsample(self, x, scale):
        if scale == 1:
            return x
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)
    def forward(self, sharp, blurred_pred, blurred_gt, depth_map):
        losses = []
        for scale in [1, 2, 4]:
            sharp_s = self.downsample(sharp, scale)
            pred_s = self.downsample(blurred_pred, scale)
            gt_s = self.downsample(blurred_gt, scale)
            depth_s = self.downsample(depth_map, scale)
            l1_loss = F.l1_loss(pred_s, gt_s)
            grad_pred_x = torch.abs(pred_s[:, :, :, 1:] - pred_s[:, :, :, :-1])
            grad_gt_x = torch.abs(gt_s[:, :, :, 1:] - gt_s[:, :, :, :-1])
            grad_pred_y = torch.abs(pred_s[:, :, 1:, :] - pred_s[:, :, :-1, :])
            grad_gt_y = torch.abs(gt_s[:, :, 1:, :] - gt_s[:, :, :-1, :])
            grad_loss = F.l1_loss(grad_pred_x, grad_gt_x) + F.l1_loss(grad_pred_y, grad_gt_y)
            blur_amount = torch.abs(pred_s - sharp_s).mean(dim=1, keepdim=True)
            depth_consistency = F.l1_loss(blur_amount, depth_s)
            losses.append((l1_loss + 0.1 * grad_loss + 0.05 * depth_consistency) / scale)
        return sum(losses) / len(losses)
class EnhancedOpticalPSFModel(nn.Module):
    def __init__(self, psf_size=31, aperture=5.6, focal_length=50.0, pixel_size=0.0064):
        super().__init__()
        self.psf_size = psf_size
        self.aperture = float(aperture)
        self.focal_length = float(focal_length)
        self.pixel_size = float(pixel_size)
        self.log_coc_scale = nn.Parameter(torch.tensor(0.0))
        self.zernike_generator = CompleteZernikePolynomials(size=256, max_order=6)
        self.zernike_coeff_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 37),
            nn.Tanh(),
        )
        self._initialize_aberration_prior()
        self.wavelengths = {0: 0.700, 1: 0.546, 2: 0.435}
    def _initialize_aberration_prior(self):
        with torch.no_grad():
            bias = self.zernike_coeff_predictor[-2].bias
            bias[3] = 0.5
            bias[4] = 0.2
    def compute_defocus_coc(self, depth_map, focus_distance_m=2.0):
        f = self.focal_length
        n = self.aperture
        u_f = focus_distance_m * 1000.0
        u_o = 500.0 + depth_map * 9500.0
        coc_mm = (f * f / (n * (u_f - f) + 1e-8)) * (torch.abs(u_o - u_f) / (u_o + 1e-8))
        coc_pix = coc_mm / (self.pixel_size + 1e-9)
        coc_pix = coc_pix * torch.exp(self.log_coc_scale)
        return coc_pix.clamp(min=0.0, max=self.psf_size)
    def generate_spatially_variant_psf(self, coc_map, position_map, channel_idx=1):
        b = coc_map.shape[0]
        avg_coc = F.adaptive_avg_pool2d(coc_map, (1, 1))
        zernike_coeffs = self.zernike_coeff_predictor(avg_coc)
        intensity_scale = (avg_coc.squeeze(-1) * 0.05).clamp(0.01, 0.5).view(b, 1)
        zernike_coeffs = zernike_coeffs * intensity_scale
        wavefront_256 = self.zernike_generator(zernike_coeffs)
        h, w = coc_map.shape[-2:]
        wavefront = F.interpolate(wavefront_256.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
        return {
            "wavefront": wavefront,
            "field_modulation": 1.0 + 0.1 * (position_map ** 2).sum(dim=1, keepdim=True),
            "distortion": 1.0,
            "zernike_coeffs": zernike_coeffs,
        }
class FFTSpatiallyVariantBlur(nn.Module):
    def __init__(self, psf_size=31, num_sigma_bases=8, sigma_min=0.2, sigma_max=12.0):
        super().__init__()
        self.psf_size = psf_size
        self.num_sigma_bases = num_sigma_bases
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_affine = nn.Conv2d(1, 1, kernel_size=1)
        nn.init.constant_(self.sigma_affine.weight, 0.3)
        nn.init.constant_(self.sigma_affine.bias, 0.5)
        sigma_bases = torch.linspace(sigma_min, sigma_max, num_sigma_bases)
        self.register_buffer("sigma_bases", sigma_bases)
        x = torch.linspace(-psf_size // 2, psf_size // 2, psf_size)
        y = torch.linspace(-psf_size // 2, psf_size // 2, psf_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        psf_list = []
        for sigma in sigma_bases:
            g = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2 + 1e-9))
            psf_list.append(g / (g.sum() + 1e-9))
        self.register_buffer("psf_bases", torch.stack(psf_list, dim=0))
    def forward(self, image, coc_map, psf_params):
        b = image.shape[0]
        sigma_map = F.softplus(self.sigma_affine(coc_map)).clamp(self.sigma_min, self.sigma_max)
        diff = sigma_map - self.sigma_bases.view(1, self.num_sigma_bases, 1, 1)
        weights = torch.exp(-(diff ** 2) / 2.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        blurred_out = torch.zeros_like(image)
        for i in range(self.num_sigma_bases):
            blurred_i = self.apply_fft_blur(image, self.psf_bases[i].expand(b, 1, -1, -1))
            blurred_out += blurred_i * weights[:, i : i + 1]
        return blurred_out
    def apply_fft_blur(self, x, psf):
        b, c, h, w = x.shape
        k = psf.shape[-1]
        pad_h, pad_w = h + k - 1, w + k - 1
        x_fft = torch.fft.rfft2(x, s=(pad_h, pad_w))
        psf_fft = torch.fft.rfft2(psf, s=(pad_h, pad_w))
        out = torch.fft.irfft2(x_fft * psf_fft, s=(pad_h, pad_w))
        start_h, start_w = (k - 1) // 2, (k - 1) // 2
        return out[:, :, start_h : start_h + h, start_w : start_w + w]
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.body(x)
class HighFreqResidualRefiner(nn.Module):
    def __init__(self, in_channels=15, base_c=64, out_scale=0.05):
        super().__init__()
        self.out_scale = out_scale
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_c, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
        )
        self.res_layers = nn.Sequential(
            ResBlock(base_c),
            ResBlock(base_c),
            ResBlock(base_c),
            ResBlock(base_c),
            nn.Dropout2d(p=0.1),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_c, base_c // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_c // 2, 3, 3, 1, 1),
            nn.Tanh(),
        )
        self.texture_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 1, 1, 0),
        )
    def forward(self, x, sharp_stage1, diagnostic_residual, maps_stage1):
        lap = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        texture_feats = self.texture_net(lap)
        conf = maps_stage1["confidence"]
        uncertain_res = diagnostic_residual * (conf + 1e-6)
        inp = torch.cat(
            [
                x,
                sharp_stage1,
                uncertain_res,
                texture_feats,
                maps_stage1["depth"],
                maps_stage1["confidence"],
                maps_stage1["sigma"],
            ],
            dim=1,
        )
        feat_init = self.input_proj(inp)
        feat_res = self.res_layers(feat_init)
        hf_residual = self.final_conv(feat_res) * self.out_scale
        return hf_residual * conf, feat_res
class FastMultiFocalDeblurNet(nn.Module):
    def __init__(self, in_channels=3, recon_base_c=24, stage1_lowpass_k=7):
        super().__init__()
        self.est = LightweightEstimator(in_channels)
        self.recon = HighResReconstructor(in_channels, base_c=recon_base_c)
        self.optical_model = EnhancedOpticalPSFModel(psf_size=31)
        self.spatial_blur = FFTSpatiallyVariantBlur(psf_size=31)
        self.physics_consistency = MultiScalePhysicsConsistency(num_scales=3)
        self.refiner = ParameterRefiner()
        self.hf_refiner = HighFreqResidualRefiner(in_channels=15, base_c=64, out_scale=0.05)
        self.stage1_lowpass_k = int(stage1_lowpass_k)
        self.register_buffer("position_encoding", self._create_position_encoding())
    def _create_position_encoding(self, h=256, w=256):
        y = torch.linspace(-1, 1, h)
        x = torch.linspace(-1, 1, w)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    def get_position_encoding(self, h, w):
        if h != self.position_encoding.shape[2] or w != self.position_encoding.shape[3]:
            dev = self.position_encoding.device
            y = torch.linspace(-1, 1, h, device=dev)
            x = torch.linspace(-1, 1, w, device=dev)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return self.position_encoding
    def _lowpass(self, img):
        k = self.stage1_lowpass_k
        if k <= 1:
            return img
        return F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x, return_params=False):
        b, c, h, w = x.shape
        d1, c1, s1, feats_s1 = self.est(x)
        residual_coarse = self.recon(x, d1, c1, s1, feats_s1)
        sharp_stage1 = torch.clamp(x + self._lowpass(residual_coarse), 0, 1)
        maps_stage1 = {"depth": d1, "confidence": c1, "sigma": s1}
        delta = self.refiner(x, sharp_stage1, maps_stage1)
        d2 = torch.clamp(d1 + delta[:, 0:1], 0, 1)
        pos_enc = self.get_position_encoding(h, w).repeat(b, 1, 1, 1)
        coc_diameter = self.optical_model.compute_defocus_coc(d2, focus_distance_m=2.0)
        psf_params = self.optical_model.generate_spatially_variant_psf(coc_diameter, pos_enc)
        reblur_pred = self.spatial_blur(sharp_stage1, coc_diameter, psf_params)
        diagnostic_residual = x - reblur_pred
        diag_cue = torch.tanh(diagnostic_residual * 15.0)
        maps_refined = {"depth": d2, "confidence": c1, "sigma": s1}
        hf_residual, hf_feat = self.hf_refiner(x, sharp_stage1, diag_cue, maps_refined)
        sharp_final = torch.clamp(sharp_stage1 + hf_residual, 0, 1)
        physics_loss = torch.zeros((), device=x.device)
        if self.training or return_params:
            physics_loss = self.physics_consistency(sharp_stage1, reblur_pred, x, d2)
        final_maps = {
            **maps_refined,
            "coc": coc_diameter,
            "physics_loss": physics_loss,
            "sharp_stage1": sharp_stage1,
            "diagnostic_residual": diagnostic_residual,
            "hf_residual": hf_residual,
            "s1_feat": feats_s1["f2"],
            "s2_feat": hf_feat,
        }
        if self.training or return_params:
            return sharp_final, reblur_pred, final_maps
        return sharp_final
def iter_images(input_dir):
    input_dir = Path(input_dir)
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
def to_chw_float(image):
    original_dtype = image.dtype
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Unsupported channel count: {image.shape}")
    if np.issubdtype(original_dtype, np.integer):
        max_value = np.iinfo(original_dtype).max
        image = image.astype(np.float32) / float(max_value)
    else:
        image = image.astype(np.float32)
        if image.max(initial=0.0) > 1.0:
            image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (2, 0, 1)), original_dtype
def output_from_tensor(tensor, dtype):
    image = tensor.detach().cpu().squeeze(0).clamp(0, 1).numpy()
    image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(dtype, np.integer):
        max_value = np.iinfo(dtype).max
        return np.round(image * max_value).clip(0, max_value).astype(dtype)
    return image.astype(np.float32)
def load_state_dict(weight_path, device):
    checkpoint = torch.load(str(weight_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if checkpoint and all(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
    return checkpoint
def pad_to_multiple(tensor, multiple=4):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor, (h, w)



def main():
    if GPU.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    image_path = Path(INPUT_PATH)
    weight_path = WEIGHT_PATH
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix: {image_path.suffix}")
    device = torch.device("cuda:0" if torch.cuda.is_available() and GPU.lower() != "cpu" else "cpu")
    model = FastMultiFocalDeblurNet(in_channels=3, recon_base_c=24).to(device)
    state_dict = load_state_dict(weight_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    out_path = image_path.with_name(f"{image_path.stem}_refocus.tiff")
    print(f"Device: {device}")
    print(f"Weight: {weight_path}")
    print(f"Input image: {image_path}")
    print(f"Output image: {out_path}")
    with torch.inference_mode():
        if image_path.suffix.lower() in {".tif", ".tiff"}:
            image = tifffile.imread(str(image_path))
        else:
            from PIL import Image
            image = np.array(Image.open(str(image_path)))
        chw, dtype = to_chw_float(image)
        tensor = torch.from_numpy(chw).unsqueeze(0).float().to(device)
        tensor, (h, w) = pad_to_multiple(tensor, multiple=4)
        pred = model(tensor)
        pred = pred[:, :, :h, :w]
        output = output_from_tensor(pred, dtype)
        tifffile.imwrite(str(out_path), output)
        print(f"Saved: {out_path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
