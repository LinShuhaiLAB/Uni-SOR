import tifffile as tiff
from skimage import filters, morphology
from scipy import optimize
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import pickle
from typing import List
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.ops as ops
from tqdm.auto import tqdm
import matplotlib
from torchvision.ops import roi_align
import ast
import matplotlib.patches as patches
import torch
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage
import pywt
import tifffile
import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from tqdm import tqdm
import re
from typing import Union, Optional
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import warnings
from tqdm import tqdm
import re
from typing import Union, Optional
from sklearn.cluster import MiniBatchKMeans
matplotlib.use('Agg')
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
class AdaptiveGradientAlignment:
    def __init__(self, output_dir, n_jobs=-1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = mp.cpu_count() * 2 if n_jobs == -1 else n_jobs
    def read_tiff(self, path):
        return tiff.imread(path)
    def invert_image(self, image):
        max_val = {'uint8': 255, 'uint16': 65535}.get(str(image.dtype), image.max())
        return max_val - image
    def to_gray(self, image):
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 4:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    @staticmethod
    def normalize(image):
        img = image.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min < 1e-10:
            return img * 0
        return (img - img_min) / (img_max - img_min)
    @staticmethod
    def multiscale_wavelet_gradient(image, scales=[1, 2]):
        image = AdaptiveGradientAlignment.normalize(image)
        gradients = []
        current_img = image.copy()
        for scale_idx in range(len(scales)):
            try:
                coeffs = pywt.dwt2(current_img, 'haar')
                cA, (cH, cV, cD) = coeffs
                wavelet_grad = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)
                gy = filters.sobel_v(current_img)
                gx = filters.sobel_h(current_img)
                sobel_grad = np.hypot(gy, gx)
                min_h = min(wavelet_grad.shape[0], sobel_grad.shape[0])
                min_w = min(wavelet_grad.shape[1], sobel_grad.shape[1])
                fused = np.add(
                    wavelet_grad[:min_h, :min_w] * 0.6,
                    sobel_grad[:min_h, :min_w] * 0.4
                )
                if fused.shape != image.shape:
                    fused = cv2.resize(fused, (image.shape[1], image.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
                gradients.append(fused)
                if scale_idx < len(scales) - 1:
                    new_h = max(current_img.shape[0] // 2, 10)
                    new_w = max(current_img.shape[1] // 2, 10)
                    current_img = cv2.resize(current_img, (new_w, new_h),
                                             interpolation=cv2.INTER_AREA)
            except:
                continue
        if not gradients:
            return np.hypot(filters.sobel_h(image), filters.sobel_v(image))
        weights = np.array([1.0, 0.7][:len(gradients)])
        weights /= weights.sum()
        result = np.zeros_like(gradients[0])
        for w, g in zip(weights, gradients):
            result += w * g
        return result
    @staticmethod
    def adaptive_gradient_thinning(gradient, percentile=85):
        gradient_norm = AdaptiveGradientAlignment.normalize(gradient)
        threshold = np.percentile(gradient_norm, percentile)
        binary = gradient_norm > threshold
        skeleton = morphology.skeletonize(binary)
        thinned = np.zeros_like(gradient, dtype=np.float32)
        if skeleton.sum() > 0:
            thinned[skeleton] = gradient[skeleton]
            thinned = cv2.GaussianBlur(thinned, (0, 0), 0.5)
        else:
            thinned = gradient
        return thinned, skeleton
    @staticmethod
    def gradient_consistency_metric(grad_a, grad_b):
        if grad_a.shape != grad_b.shape:
            grad_b = cv2.resize(grad_b, (grad_a.shape[1], grad_a.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        flat_a = grad_a.ravel()
        flat_b = grad_b.ravel()
        mean_a, mean_b = flat_a.mean(), flat_b.mean()
        std_a, std_b = flat_a.std(), flat_b.std()
        mag_corr = 0.0
        if std_a > 0 and std_b > 0:
            mag_corr = np.mean((flat_a - mean_a) * (flat_b - mean_b)) / (std_a * std_b)
            mag_corr = 0.0 if np.isnan(mag_corr) else np.clip(mag_corr, -1, 1)
        angle_consistency = 0.0
        try:
            gy_a, gx_a = cv2.Sobel(grad_a, cv2.CV_64F, 0, 1), cv2.Sobel(grad_a, cv2.CV_64F, 1, 0)
            gy_b, gx_b = cv2.Sobel(grad_b, cv2.CV_64F, 0, 1), cv2.Sobel(grad_b, cv2.CV_64F, 1, 0)
            angle_diff = np.abs(np.arctan2(gy_a, gx_a) - np.arctan2(gy_b, gx_b))
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            angle_consistency = 1 - np.mean(angle_diff) / np.pi
            angle_consistency = 0.0 if np.isnan(angle_consistency) else angle_consistency
        except:
            pass
        return 0.5 * mag_corr + 0.5 * angle_consistency
    def extract_patches(self, image_a, image_b, patch_size=100, expand_ratio=1.2):
        h, w = image_a.shape[:2]
        n_h, n_w = h // patch_size, w // patch_size
        expand_size = int(patch_size * expand_ratio)
        half_expand = (expand_size - patch_size) // 2
        patches_a, patches_b, positions = [], [], []
        for i in range(n_h):
            for j in range(n_w):
                y1, y2 = i * patch_size, (i + 1) * patch_size
                x1, x2 = j * patch_size, (j + 1) * patch_size
                y1b = max(0, y1 - half_expand)
                y2b = min(h, y2 + half_expand)
                x1b = max(0, x1 - half_expand)
                x2b = min(w, x2 + half_expand)
                patch_a = image_a[y1:y2, x1:x2].copy()
                patch_b = image_b[y1b:y2b, x1b:x2b].copy()
                if patch_a.size > 0 and patch_b.size > 0:
                    patches_a.append(patch_a)
                    patches_b.append(patch_b)
                    positions.append({'idx': (i, j)})
        return patches_a, patches_b, positions
    def align_patch_worker(self, args):
        patch_a, patch_b, max_shift, max_angle = args
        grad_a = self.multiscale_wavelet_gradient(patch_a)
        grad_a_thin, _ = self.adaptive_gradient_thinning(grad_a)
        grad_a_thin = np.ascontiguousarray(grad_a_thin.astype(np.float32))
        grad_b = self.multiscale_wavelet_gradient(patch_b)
        grad_b_thin_base, _ = self.adaptive_gradient_thinning(grad_b)
        grad_b_thin_base = np.ascontiguousarray(grad_b_thin_base.astype(np.float32))
        flat_a = grad_a_thin.ravel()
        mean_a = float(flat_a.mean())
        std_a = float(flat_a.std())
        gy_a = cv2.Sobel(grad_a_thin, cv2.CV_32F, 0, 1, ksize=3)
        gx_a = cv2.Sobel(grad_a_thin, cv2.CV_32F, 1, 0, ksize=3)
        angle_a = np.arctan2(gy_a, gx_a)
        h, w = grad_a_thin.shape
        cy, cx = h / 2.0, w / 2.0
        DEG2RAD = np.pi / 180.0
        def fast_metric(grad_b_aligned: np.ndarray) -> float:
            flat_b = grad_b_aligned.ravel()
            mean_b = float(flat_b.mean())
            std_b = float(flat_b.std())
            mag_corr = 0.0
            if std_a > 0 and std_b > 0:
                mag_corr = float(np.mean((flat_a - mean_a) * (flat_b - mean_b)) / (std_a * std_b))
                if np.isnan(mag_corr):
                    mag_corr = 0.0
                else:
                    mag_corr = float(np.clip(mag_corr, -1, 1))
            angle_consistency = 0.0
            try:
                gy_b = cv2.Sobel(grad_b_aligned, cv2.CV_32F, 0, 1, ksize=3)
                gx_b = cv2.Sobel(grad_b_aligned, cv2.CV_32F, 1, 0, ksize=3)
                angle_b = np.arctan2(gy_b, gx_b)
                angle_diff = np.abs(angle_a - angle_b)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                angle_consistency = 1.0 - float(np.mean(angle_diff) / np.pi)
                if np.isnan(angle_consistency):
                    angle_consistency = 0.0
            except:
                pass
            return 0.5 * mag_corr + 0.5 * angle_consistency
        best_score = -np.inf
        best_params = [0.0, 0.0, 0.0]
        def objective(params):
            nonlocal best_score, best_params
            dy, dx, angle = params
            rad = angle * DEG2RAD
            cos_a = float(np.cos(rad))
            sin_a = float(np.sin(rad))
            M = np.array([
                [cos_a, -sin_a, dx + cx - cx * cos_a + cy * sin_a],
                [sin_a, cos_a, dy + cy - cx * sin_a - cy * cos_a]
            ], dtype=np.float32)
            grad_b_aligned = cv2.warpAffine(
                grad_b_thin_base, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            score = fast_metric(grad_b_aligned)
            if score > best_score:
                best_score = score
                best_params = [float(dy), float(dx), float(angle)]
            return -score
        for angle in np.linspace(-max_angle, max_angle, 5):
            objective([0.0, 0.0, float(angle)])
        try:
            optimize.minimize(
                objective, best_params, method='Powell',
                bounds=[(-max_shift, max_shift), (-max_shift, max_shift), (-max_angle, max_angle)],
                options={'maxiter': 30, 'disp': False}
            )
        except:
            pass
        return best_params, best_score
    def compute_global_transform(self, results):
        if not results:
            return {'dy': 0.0, 'dx': 0.0, 'angle': 0.0}
        transforms = np.array([r['transform'] for r in results])
        scores = np.array([r['score'] for r in results])
        valid_mask = scores > 0.3
        if valid_mask.sum() > 0:
            transforms = transforms[valid_mask]
            scores = scores[valid_mask]
        weights = scores / (scores.sum() + 1e-10)
        weighted_avg = np.average(transforms, axis=0, weights=weights)
        median_vals = np.median(transforms, axis=0)
        final = 0.7 * weighted_avg + 0.3 * median_vals
        return {
            'dy': float(final[0]),
            'dx': float(final[1]),
            'angle': float(final[2])
        }
    def save_params_txt(self, global_params, source_file, target_file, canvas_shape):
        output_file = self.output_dir / 'alignment_params.txt'
        h, w = canvas_shape[:2]
        rot_center_x = w / 2
        rot_center_y = h / 2
        content = f"""Alignment Parameters (Stage 2 - Patch-based Refinement)
===================================================
Target (MSI): {Path(target_file).name}
Source (HE):  {Path(source_file).name}
Translation_X: {global_params['dx']}
Translation_Y: {global_params['dy']}
Rotation_Deg:  {global_params['angle']}
Rotation_Center_X: {rot_center_x}
Rotation_Center_Y: {rot_center_y}
Note: Rotation center is canvas center (MSI image center).
Apply Translation first, then Rotation around the specified center.
"""
        with open(output_file, 'w') as f:
            f.write(content)
    def process(self, path_a, path_b, patch_size=100, max_shift=5, max_angle=10):
        img_a = self.read_tiff(path_a)
        img_b = self.read_tiff(path_b)
        img_b = self.invert_image(img_b)
        img_a_gray = self.to_gray(img_a)
        img_b_gray = self.to_gray(img_b)
        if img_a_gray.shape != img_b_gray.shape:
            img_b_gray = cv2.resize(img_b_gray,
                                    (img_a_gray.shape[1], img_a_gray.shape[0]),
                                    interpolation=cv2.INTER_AREA)
        patches_a, patches_b, positions = self.extract_patches(
            img_a_gray, img_b_gray, patch_size=patch_size
        )
        tasks = [(pa, pb, max_shift, max_angle)
                 for pa, pb in zip(patches_a, patches_b)]
        results = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(self.align_patch_worker, task): i
                       for i, task in enumerate(tasks)}
            with tqdm(total=len(futures), desc="Alignment", ncols=80) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        transform_params, score = future.result()
                        results.append({
                            'position': positions[idx]['idx'],
                            'transform': transform_params,
                            'score': score
                        })
                        pbar.set_postfix({'score': f'{score:.3f}'})
                    except Exception as e:
                        print(f"\nWarning: Patch {idx} failed: {e}")
                    pbar.update(1)
        global_params = self.compute_global_transform(results)
        print(f"  Translation_X: {global_params['dx']:.3f} px")
        print(f"  Translation_Y: {global_params['dy']:.3f} px")
        print(f"  Rotation_Deg:  {global_params['angle']:.3f}°")
        self.save_params_txt(global_params, path_b, path_a, img_a.shape)
        return global_params
def generate_unified_mapping_csv(root_dir, output_csv_path):
    subfolders_map = {
        "HE_Feature": "HE_Feature",
        "HE_Heatmap": "HE_Heatmap",
        "original": "original",
        "sampling": "sampling"
    }
    mapping_data = {}
    print(f"Scanning root directory: {root_dir}")
    for column_name, folder_name in subfolders_map.items():
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Directory does not exist - {folder_path}")
            continue
        files = os.listdir(folder_path)
        print(f"  -> Processing {folder_name}: found {len(files)} files")
        for file_name in files:
            if file_name.startswith('.'):
                continue
            full_path = os.path.join(folder_path, file_name)
            file_stem = Path(file_name).stem
            unique_id = None
            try:
                if column_name == "original":
                    if "_original_" in file_stem:
                        unique_id = file_stem.split("_original_")[1]
                elif column_name == "sampling":
                    if "_sampling_" in file_stem:
                        unique_id = file_stem.split("_sampling_")[1]
                elif column_name == "HE_Feature":
                    if file_stem.startswith("HE_Feature_"):
                        unique_id = file_stem.replace("HE_Feature_", "", 1)
                elif column_name == "HE_Heatmap":
                    if file_stem.startswith("HE_Heatmap_"):
                        unique_id = file_stem.replace("HE_Heatmap_", "", 1)
            except Exception as e:
                print(f"Error parsing filename: {file_name} - {e}")
                continue
            if unique_id:
                if unique_id not in mapping_data:
                    mapping_data[unique_id] = {}
                mapping_data[unique_id][column_name] = full_path
    print("Building DataFrame...")
    df = pd.DataFrame.from_dict(mapping_data, orient='index')
    desired_columns = ["original", "sampling", "HE_Feature", "HE_Heatmap"]
    df = df.reindex(columns=desired_columns)
    initial_count = len(df)
    df.dropna(inplace=True)
    final_count = len(df)
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Filtered out {dropped_count} IDs that were missing one or more files.")
    if df.empty:
        print("Error: No complete datasets found (all IDs are missing at least one file).")
        return
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\nSuccess! Mapping file saved to: {output_csv_path}")
        print(f"Total complete entries: {len(df)}")
        print("\nPreview (First 3 rows):")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"Error saving CSV file: {e}")
def parse_coordinates(col_names):
    import re
    coords = []
    valid_indices = []
    pattern = re.compile(r'-(\d+)-(\d+)$')
    for idx, name in enumerate(col_names):
        match = pattern.search(str(name))
        if match:
            coords.append((int(match.group(1)), int(match.group(2))))
            valid_indices.append(idx)
    return coords, valid_indices
def normalize_to_01(matrix: np.ndarray) -> np.ndarray:
    if np.sum(matrix) == 0:
        return np.zeros_like(matrix, dtype=np.float32)
    valid_mask = matrix > 0
    valid_data = matrix[valid_mask]
    if len(valid_data) == 0:
        return np.zeros_like(matrix, dtype=np.float32)
    v_min = valid_data.min()
    v_max = valid_data.max()
    normalized = np.zeros_like(matrix, dtype=np.float32)
    if v_max > v_min:
        normalized[valid_mask] = (matrix[valid_mask] - v_min) / (v_max - v_min)
    elif v_max == v_min and v_max > 0:
        normalized[valid_mask] = 1.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
class GradientDetector:
    def __init__(self, border_erosion_iterations: int = 1):
        self.border_erosion_iterations = border_erosion_iterations
    def _get_signal_mask(self, image: np.ndarray) -> np.ndarray:
        has_signal = (image > 0).astype(np.uint8)
        if np.sum(has_signal) == 0: return np.zeros_like(has_signal)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(has_signal, kernel, iterations=self.border_erosion_iterations)
    def detect(self, image: np.ndarray) -> np.ndarray: raise NotImplementedError
class SobelDetector(GradientDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        internal_mask = self._get_signal_mask(image)
        if np.sum(internal_mask) == 0: return np.zeros_like(image, dtype=np.float32)
        grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y) * internal_mask
class LaplacianDetector(GradientDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        internal_mask = self._get_signal_mask(image)
        if np.sum(internal_mask) == 0: return np.zeros_like(image, dtype=np.float32)
        laplacian = cv2.Laplacian(image.astype(np.float64), cv2.CV_64F, ksize=3)
        return np.abs(laplacian).astype(np.float32) * internal_mask
class WaveletSobelDetector(GradientDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        internal_mask = self._get_signal_mask(image)
        if np.sum(internal_mask) == 0: return np.zeros_like(image, dtype=np.float32)
        coeffs = pywt.dwt2(image, 'db4')
        cA, (cH, cV, cD) = coeffs
        denoised = pywt.idwt2((cA, (cH * 0.8, cV * 0.8, cD * 0.8)), 'db4')
        if denoised.shape != image.shape:
            denoised = cv2.resize(denoised, (image.shape[1], image.shape[0]))
        grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y) * internal_mask
class CannyDetector(GradientDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        internal_mask = self._get_signal_mask(image)
        if np.sum(internal_mask) == 0: return np.zeros_like(image, dtype=np.float32)
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        valid_pixels = img_norm[img_norm > 0]
        if len(valid_pixels) == 0: return np.zeros_like(image, dtype=np.float32)
        median_val = np.median(valid_pixels)
        lower = int(max(0, 0.7 * median_val))
        upper = int(min(255, 1.3 * median_val))
        return cv2.Canny(img_norm, lower, upper).astype(np.float32) * internal_mask
class EntropyDetector(GradientDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        internal_mask = self._get_signal_mask(image)
        if np.sum(internal_mask) == 0: return np.zeros_like(image, dtype=np.float32)
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        local_entropy = rank_entropy(img_norm, disk(2))
        return local_entropy.astype(np.float32) * internal_mask
class FlowRateDetector(GradientDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        internal_mask = self._get_signal_mask(image)
        if np.sum(internal_mask) == 0: return np.zeros_like(image, dtype=np.float32)
        img_float = image.astype(np.float32)
        img_smooth = cv2.GaussianBlur(img_float, (3, 3), 1.0)
        flow_rate = np.abs(img_float - img_smooth)
        grad_x = cv2.Sobel(flow_rate, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(flow_rate, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y) * internal_mask
class ContourDetector(GradientDetector):
    def detect(self, image: np.ndarray) -> np.ndarray:
        internal_mask = self._get_signal_mask(image)
        if np.sum(internal_mask) == 0: return np.zeros_like(image, dtype=np.float32)
        img_smooth = cv2.GaussianBlur(image, (3, 3), 0.5)
        valid_pixels = img_smooth[img_smooth > 0]
        if len(valid_pixels) == 0: return np.zeros_like(image, dtype=np.float32)
        v_min, v_max = valid_pixels.min(), valid_pixels.max()
        if v_max <= v_min: return np.zeros_like(image, dtype=np.float32)
        levels = np.linspace(v_min, v_max, 12)[1:-1]
        edge_map = np.zeros_like(image, dtype=np.float32)
        for level in levels:
            binary = (img_smooth > level).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) > 3:
                    cv2.drawContours(edge_map, [contour], -1, 1.0, 1)
        return edge_map * internal_mask
GRADIENT_DETECTORS = {
    'sobel': SobelDetector, 'laplacian': LaplacianDetector, 'wavelet_sobel': WaveletSobelDetector,
    'canny': CannyDetector, 'entropy': EntropyDetector, 'flow_rate': FlowRateDetector, 'contour': ContourDetector,
}
def save_tiff_heatmap(
        normalized_matrix: np.ndarray,
        output_path: Path,
        target_w: int,
        target_h: int,
        target_dpi: int,
        cmap
) -> None:
    if np.sum(normalized_matrix) == 0:
        print(f"    Warning {output_path.name} No data.")
        return
    processed_matrix = np.clip(normalized_matrix, 0.0, 1.0)
    rgba_float = cmap(processed_matrix)
    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    alpha_mask = (processed_matrix > 0).astype(np.uint8) * 255
    rgba_uint8[:, :, 3] = alpha_mask
    img_low_res = Image.fromarray(rgba_uint8, mode='RGBA')
    img_final = img_low_res.resize((target_w, target_h), resample=Image.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_final.save(str(output_path), format='TIFF', compression='tiff_deflate', dpi=(target_dpi, target_dpi))
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
def apply_adaptive_threshold(grad_mag: np.ndarray, max_pixel_ratio: float) -> np.ndarray:
    valid_pixels = grad_mag[grad_mag > 0]
    if len(valid_pixels) == 0: return np.zeros_like(grad_mag, dtype=np.uint8)
    total_pixels = np.sum(grad_mag > 0)
    max_edge_pixels = int(total_pixels * max_pixel_ratio)
    for percentile in range(95, 60, -1):
        threshold = np.percentile(valid_pixels, percentile)
        mask = (grad_mag > threshold).astype(np.uint8)
        if np.sum(mask) <= max_edge_pixels: return mask
    return (grad_mag > np.percentile(valid_pixels, 60)).astype(np.uint8)
def apply_frequency_threshold(freq_map: np.ndarray, top_ratio: float) -> np.ndarray:
    valid_freqs = freq_map[freq_map > 0]
    if len(valid_freqs) == 0: return np.zeros_like(freq_map, dtype=np.float32)
    keep_pixels = max(1, int(len(valid_freqs) * top_ratio))
    threshold_freq = np.sort(valid_freqs)[::-1][min(keep_pixels - 1, len(valid_freqs) - 1)]
    return np.where(freq_map >= threshold_freq, freq_map, 0).astype(np.float32)
def process_msi_file(
        input_file: str,
        target_mz: Optional[float] = None,
        output_first_row_tiff: bool = True,
        output_gradient_tiff: bool = True,
        gradient_methods: Optional[List[str]] = None,
        msi_resolution_microns: float = 50.0,
        target_dpi: int = 1000,
        first_row_colormap: str = 'viridis',
        max_edge_pixel_ratio: float = 0.10,
        final_frequency_top_ratio: float = 0.20,
        border_erosion_iterations: int = 1
) -> None:
    if gradient_methods is None:
        gradient_methods = ['sobel', 'laplacian', 'wavelet_sobel', 'canny', 'entropy', 'flow_rate', 'contour']
    file_path = Path(input_file)
    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
    except FileNotFoundError:
        print(f"  ❌ ERROR: Cannot find {file_path}")
        return
    data_start_col = 22
    coords, valid_col_indices = parse_coordinates(df.columns[data_start_col:])
    if not coords:
        print("  ❌ ERROR: Cannot find coordinates.")
        return
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    width, height = max(xs) + 1, max(ys) + 1
    mz_values = df.iloc[:, 1].values
    pixel_data = df.iloc[:, [x + data_start_col for x in valid_col_indices]].values
    num_mz = len(mz_values)
    if target_mz is None:
        target_mz_index = 0
    else:
        mz_numeric = pd.to_numeric(mz_values, errors='coerce')
        valid_mz_mask = ~np.isnan(mz_numeric)
        if not np.any(valid_mz_mask):
            print(f"  ❌ ERROR: no m/z")
            return
        valid_mz = mz_numeric[valid_mz_mask]
        valid_indices = np.where(valid_mz_mask)[0]
        closest_idx = valid_indices[np.argmin(np.abs(valid_mz - target_mz))]
        target_mz_index = closest_idx
    first_row_matrix = np.zeros((height, width), dtype=np.float32)
    first_row_matrix[ys, xs] = np.nan_to_num(pd.to_numeric(pixel_data[target_mz_index, :], errors='coerce'))
    gradient_frequency_maps = {method: np.zeros((height, width), dtype=np.float32) for method in gradient_methods}
    for i in range(num_mz):
        img = np.zeros((height, width), dtype=np.float32)
        img[ys, xs] = np.nan_to_num(pd.to_numeric(pixel_data[i, :], errors='coerce'))
        for method in gradient_methods:
            try:
                detector = GRADIENT_DETECTORS[method](border_erosion_iterations)
                grad_mag = detector.detect(img)
                mask = apply_adaptive_threshold(grad_mag, max_edge_pixel_ratio)
                gradient_frequency_maps[method][mask > 0] += 1
            except Exception as e:
                pass
    print("")
    for method in gradient_methods:
        gradient_frequency_maps[method] = apply_frequency_threshold(gradient_frequency_maps[method],
                                                                    final_frequency_top_ratio)
    normalized_gradient_maps = {}
    for method in gradient_methods:
        normalized_gradient_maps[method] = normalize_to_01(gradient_frequency_maps[method])
    output_dir = file_path.parent
    phys_w_inch = (width * msi_resolution_microns) / 25400.0
    target_w = int(round(phys_w_inch * target_dpi))
    target_h = int(round(height * (target_w / width)))
    pkl_data = {
        'first_mz': {'raw': first_row_matrix},
        'gradient_results': {
            m: {'raw_counts': gradient_frequency_maps[m], 'normalized': normalized_gradient_maps[m]}
            for m in gradient_methods
        }
    }
    pkl_path = output_dir / f"{file_path.stem}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    if output_first_row_tiff:
        first_row_norm = normalize_to_01(first_row_matrix)
        cmap_first = plt.get_cmap(first_row_colormap)
        save_tiff_heatmap(first_row_norm, output_dir / f"{file_path.stem}_first_mz.tiff",
                          target_w, target_h, target_dpi, cmap_first)
    if output_gradient_tiff:
        cmap_gray = plt.get_cmap('gray')
        for method, norm_map in normalized_gradient_maps.items():
            save_tiff_heatmap(norm_map, output_dir / f"{file_path.stem}_gradient_{method}_density.tiff",
                              target_w, target_h, target_dpi, cmap_gray)
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
        pad_size = 8
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
        final_img_blocky = apply_grid_median_smoothing(final_img_pixel, h_lines_list, v_lines_list)
        return final_img_blocky, coarse_img
def load_image_gray_optimized(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return None, None
    try:
        if path.endswith('.npy'):
            img = np.load(path)
            if img.ndim == 3: img = np.mean(img, axis=2)
            min_v, max_v = np.min(img), np.max(img)
            denom = max_v - min_v
            if denom == 0: denom = 1e-8
            img_float = (img - min_v) / denom
            img_uint8 = (img_float * 255).astype(np.uint8)
            return img_uint8, img_float
        file_data = np.fromfile(path, dtype=np.uint8)
        img_uint8 = cv2.imdecode(file_data, cv2.IMREAD_GRAYSCALE)
        if img_uint8 is None: return None, None
        img_float = img_uint8.astype(np.float32) / 255.0
        return img_uint8, img_float
    except:
        return None, None
def fast_mode_val(patch):
    if patch.size == 0: return 0
    return np.argmax(np.bincount(patch.ravel(), minlength=256))
def extract_patches_raw(img_data, grid_x, grid_y, method='raw'):
    n_rows = len(grid_y) - 1
    n_cols = len(grid_x) - 1
    h_img, w_img = img_data.shape
    if method == 'mode':
        feature_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        for r in range(n_rows):
            y1, y2 = grid_y[r], grid_y[r + 1]
            if y1 >= h_img: continue
            for c in range(n_cols):
                x1, x2 = grid_x[c], grid_x[c + 1]
                if x1 >= w_img: continue
                patch = img_data[y1:y2, x1:x2]
                if patch.size == 0: continue
                m = fast_mode_val(patch)
                feature_map[r, c] = float(m) / 255.0
        return feature_map
    elif method == 'raw':
        raw_patches = {}
        for r in range(n_rows):
            y1, y2 = grid_y[r], grid_y[r + 1]
            if y1 >= h_img: continue
            for c in range(n_cols):
                x1, x2 = grid_x[c], grid_x[c + 1]
                if x1 >= w_img: continue
                patch = img_data[y1:y2, x1:x2]
                if patch.size == 0: continue
                raw_patches[(r, c)] = patch.copy()
        return raw_patches
    return None
def process_single_row(row_tuple):
    index, sid, p_orig, p_samp, p_heat, grid_x, grid_y = row_tuple
    try:
        orig_u8, _ = load_image_gray_optimized(p_orig)
        if orig_u8 is None: return None
        map_orig = extract_patches_raw(orig_u8, grid_x, grid_y, method='mode')
        _, samp_f = load_image_gray_optimized(p_samp)
        if samp_f is None: return None
        dict_samp = extract_patches_raw(samp_f, grid_x, grid_y, method='raw')
        global_samp = float(np.mean(samp_f))
        _, heat_f = load_image_gray_optimized(p_heat)
        if heat_f is None: return None
        dict_heat = extract_patches_raw(heat_f, grid_x, grid_y, method='raw')
        global_heat = float(np.mean(heat_f))
        return {
            "id": sid,
            "res": (grid_x, grid_y, map_orig, dict_samp, global_samp, dict_heat, global_heat)
        }
    except:
        return None
def visualize_grid_verification(row_data, output_png_path):
    print(f"\nGenerating Precision Grid Visualization: {output_png_path} ...")
    sid = str(row_data['ID'])
    grid_x = ast.literal_eval(row_data['grid_x'])
    grid_y = ast.literal_eval(row_data['grid_y'])
    img_orig, _ = load_image_gray_optimized(row_data['original'])
    _, img_samp = load_image_gray_optimized(row_data['sampling'])
    _, img_heat = load_image_gray_optimized(row_data['HE_Heatmap'])
    if img_orig is None: return
    tgt_r, tgt_c = 0, 0
    if len(grid_y) < 2 or len(grid_x) < 2: return
    y1, y2 = grid_y[tgt_r], grid_y[tgt_r + 1]
    x1, x2 = grid_x[tgt_c], grid_x[tgt_c + 1]
    patch_orig = img_orig[y1:y2, x1:x2]
    patch_samp = img_samp[y1:y2, x1:x2]
    patch_heat = img_heat[y1:y2, x1:x2]
    mode_val = fast_mode_val(patch_orig)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plt.suptitle(f"Grid Alignment & Encoding Verification (ID: {sid})", fontsize=18)
    h, w = img_orig.shape
    def draw_grid_lines(ax, gx, gy, img_h, img_w):
        for x in gx:
            if 0 <= x <= img_w:
                ax.axvline(x, color='lime', linestyle='-', linewidth=0.8, alpha=0.9)
        for y in gy:
            if 0 <= y <= img_h:
                ax.axhline(y, color='lime', linestyle='-', linewidth=0.8, alpha=0.9)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2.5, edgecolor='red', facecolor='none', zorder=10)
        ax.add_patch(rect)
    axes[0, 0].imshow(img_orig, cmap='gray', vmin=0, vmax=255, extent=[0, w, h, 0])
    draw_grid_lines(axes[0, 0], grid_x, grid_y, h, w)
    axes[0, 0].set_title("Original (Green Grid = Pixel Boundaries)", fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel("Global Y Pixels")
    axes[0, 1].imshow(img_samp, cmap='viridis', vmin=0, vmax=1, extent=[0, w, h, 0])
    draw_grid_lines(axes[0, 1], grid_x, grid_y, h, w)
    axes[0, 1].set_title("Sampling (Green Grid = Pixel Boundaries)", fontsize=14, fontweight='bold')
    axes[0, 2].imshow(img_heat, cmap='viridis', vmin=0, vmax=1, extent=[0, w, h, 0])
    draw_grid_lines(axes[0, 2], grid_x, grid_y, h, w)
    axes[0, 2].set_title("HE Heatmap (Green Grid = Pixel Boundaries)", fontsize=14, fontweight='bold')
    ph, pw = patch_orig.shape
    im1 = axes[1, 0].imshow(patch_orig, cmap='gray', vmin=0, vmax=255, extent=[0, pw, ph, 0])
    axes[1, 0].set_title(f"Patch [0,0] Result: Mode = {mode_val}", fontsize=14, color='blue', fontweight='bold')
    axes[1, 0].set_xlabel("Local X Pixels")
    axes[1, 0].set_ylabel("Local Y Pixels")
    axes[1, 0].grid(color='white', linestyle=':', linewidth=0.5, alpha=0.3)
    fig.colorbar(im1, ax=axes[1, 0], label='Label (0-255)')
    im2 = axes[1, 1].imshow(patch_samp, cmap='viridis', vmin=0, vmax=1, extent=[0, pw, ph, 0])
    axes[1, 1].set_title(f"Patch [0,0] Result: Raw Matrix", fontsize=14, color='green', fontweight='bold')
    axes[1, 1].set_xlabel("Local X Pixels")
    fig.colorbar(im2, ax=axes[1, 1], label='Intensity (0-1)')
    im3 = axes[1, 2].imshow(patch_heat, cmap='viridis', vmin=0, vmax=1, extent=[0, pw, ph, 0])
    axes[1, 2].set_title(f"Patch [0,0] Result: Raw Matrix", fontsize=14, color='green', fontweight='bold')
    axes[1, 2].set_xlabel("Local X Pixels")
    fig.colorbar(im3, ax=axes[1, 2], label='Intensity (0-1)')
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150)
    plt.close()
    print(f"Success! Saved to {output_png_path}")
def main_process(csv_path, output_pth_path, output_png_path):
    if not os.path.exists(csv_path):
        print("CSV file not found.")
        return
    print(f"Reading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path, encoding='gbk')
    tasks = []
    valid_indices = []
    print("Preparing tasks...")
    for index, row in df.iterrows():
        try:
            gx = ast.literal_eval(row['grid_x'])
            gy = ast.literal_eval(row['grid_y'])
            if len(gx) >= 2 and len(gy) >= 2:
                tasks.append((index, str(row['ID']), row['original'], row['sampling'], row['HE_Heatmap'], gx, gy))
                valid_indices.append(index)
        except:
            continue
    max_workers = min(4, os.cpu_count())
    print(f"Starting encoding with {max_workers} workers...")
    dataset_encoded = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_row, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            if res is not None:
                sid = res['id']
                gx, gy, map_orig, dict_samp, glob_samp, dict_heat, glob_heat = res['res']
                import torch
                tensor_orig = torch.from_numpy(map_orig).type(torch.float16)
                patches_samp = {k: torch.from_numpy(v).type(torch.float16) for k, v in dict_samp.items()}
                patches_heat = {k: torch.from_numpy(v).type(torch.float16) for k, v in dict_heat.items()}
                dataset_encoded[sid] = {
                    "grid": {
                        "x": torch.tensor(gx, dtype=torch.int16),
                        "y": torch.tensor(gy, dtype=torch.int16)
                    },
                    "original": tensor_orig,
                    "sampling": {
                        "patches": patches_samp,
                        "global": glob_samp
                    },
                    "he_heatmap": {
                        "patches": patches_heat,
                        "global": glob_heat
                    }
                }
    if dataset_encoded:
        print(f"\nSaving {len(dataset_encoded)} samples to {output_pth_path}...")
        import torch
        torch.save(dataset_encoded, output_pth_path)
        print("Save Success!")
        if valid_indices:
            visualize_grid_verification(df.loc[valid_indices[0]], output_png_path)
    else:
        print("❌ Error: No samples processed.")
def load_alignment_params(params_path):
    params = {}
    with open(params_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Translation_X:' in line:
                params['tx'] = float(line.split(':')[1].strip())
            elif 'Translation_Y:' in line:
                params['ty'] = float(line.split(':')[1].strip())
            elif 'Rotation_Deg:' in line:
                params['rot'] = float(line.split(':')[1].strip())
            elif 'Rotation_Center_X:' in line:
                params['cx'] = float(line.split(':')[1].strip())
            elif 'Rotation_Center_Y:' in line:
                params['cy'] = float(line.split(':')[1].strip())
    rot_center = (params['cx'], params['cy']) if 'cx' in params and 'cy' in params else None
    return params['tx'], params['ty'], params['rot'], rot_center
def get_coarse_alignment_matrix(he_path, desi_path, params_path):
    tx, ty, rot, rot_center = load_alignment_params(params_path)
    he_img = tifffile.imread(he_path)
    desi_img = tifffile.imread(desi_path)
    if rot_center is None:
        canvas_h, canvas_w = desi_img.shape[:2]
        rot_center = (canvas_w / 2, canvas_h / 2)
    M_T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    M_R_2x3 = cv2.getRotationMatrix2D(rot_center, rot, 1.0)
    M_R = np.vstack([M_R_2x3, [0, 0, 1]])
    M_final = M_R @ M_T
    canvas_h, canvas_w = desi_img.shape[:2]
    he_aligned = cv2.warpAffine(he_img, M_final[:2, :], (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255, 0))
    return M_final, he_aligned, desi_img
def get_valid_mask(img, erosion_size=1):
    if img.shape[2] == 4:
        mask = img[:, :, 3] > 0
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray < 240
    return mask
def extract_edges_multi_method(img_gray, valid_mask,
                               methods=['sobel', 'laplacian', 'wavelet_sobel', 'canny',
                                        'entropy', 'flow_rate', 'contour'],
                               max_edge_pixel_ratio=0.1):
    edges_dict = {}
    valid_pixel_count = np.sum(valid_mask)
    threshold_count = int(valid_pixel_count * max_edge_pixel_ratio)
    if 'sobel' in methods:
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel_masked = sobel * valid_mask
        valid_values = sobel_masked[valid_mask]
        if len(valid_values) > 0:
            thresh = np.sort(valid_values)[-min(threshold_count, len(valid_values))]
            edges_dict['sobel'] = ((sobel_masked >= thresh) & valid_mask).astype(np.uint8)
        else:
            edges_dict['sobel'] = np.zeros_like(valid_mask, dtype=np.uint8)
    if 'laplacian' in methods:
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian_masked = laplacian * valid_mask
        valid_values = laplacian_masked[valid_mask]
        if len(valid_values) > 0:
            thresh = np.sort(valid_values)[-min(threshold_count, len(valid_values))]
            edges_dict['laplacian'] = ((laplacian_masked >= thresh) & valid_mask).astype(np.uint8)
        else:
            edges_dict['laplacian'] = np.zeros_like(valid_mask, dtype=np.uint8)
    if 'wavelet_sobel' in methods:
        coeffs = pywt.dwt2(img_gray, 'haar')
        cA, (cH, cV, cD) = coeffs
        wavelet_edge = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)
        h, w = img_gray.shape
        wavelet_edge = cv2.resize(wavelet_edge, (w, h))
        wavelet_masked = wavelet_edge * valid_mask
        valid_values = wavelet_masked[valid_mask]
        if len(valid_values) > 0:
            thresh = np.sort(valid_values)[-min(threshold_count, len(valid_values))]
            edges_dict['wavelet_sobel'] = ((wavelet_masked >= thresh) & valid_mask).astype(np.uint8)
        else:
            edges_dict['wavelet_sobel'] = np.zeros_like(valid_mask, dtype=np.uint8)
    if 'canny' in methods:
        median_val = np.median(img_gray[valid_mask])
        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))
        canny = cv2.Canny(img_gray, lower, upper)
        edges_dict['canny'] = ((canny > 0) & valid_mask).astype(np.uint8)
    if 'entropy' in methods:
        from skimage.filters.rank import entropy as rank_entropy
        from skimage.morphology import disk
        entropy_img = rank_entropy(img_gray, disk(5))
        entropy_masked = entropy_img * valid_mask
        valid_values = entropy_masked[valid_mask]
        if len(valid_values) > 0:
            thresh = np.sort(valid_values)[-min(threshold_count, len(valid_values))]
            edges_dict['entropy'] = ((entropy_masked >= thresh) & valid_mask).astype(np.uint8)
        else:
            edges_dict['entropy'] = np.zeros_like(valid_mask, dtype=np.uint8)
    if 'flow_rate' in methods:
        gx = ndimage.sobel(img_gray.astype(float), axis=1)
        gy = ndimage.sobel(img_gray.astype(float), axis=0)
        flow = np.sqrt(gx ** 2 + gy ** 2)
        flow_masked = flow * valid_mask
        valid_values = flow_masked[valid_mask]
        if len(valid_values) > 0:
            thresh = np.sort(valid_values)[-min(threshold_count, len(valid_values))]
            edges_dict['flow_rate'] = ((flow_masked >= thresh) & valid_mask).astype(np.uint8)
        else:
            edges_dict['flow_rate'] = np.zeros_like(valid_mask, dtype=np.uint8)
    if 'contour' in methods:
        img_masked = img_gray.copy()
        img_masked[~valid_mask] = 0
        _, binary = cv2.threshold(img_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary & valid_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(img_gray, dtype=np.uint8)
        cv2.drawContours(contour_img, contours, -1, 1, 1)
        edges_dict['contour'] = (contour_img & valid_mask).astype(np.uint8)
    return edges_dict
def integrate_edges(edges_dict, final_frequency_top_ratio=0.3):
    frequency_map = np.zeros_like(list(edges_dict.values())[0], dtype=np.float32)
    for edge in edges_dict.values():
        frequency_map += edge
    if frequency_map.max() > 0:
        frequency_map = frequency_map / frequency_map.max()
    threshold = np.percentile(frequency_map[frequency_map > 0], (1 - final_frequency_top_ratio) * 100)
    integrated_edge = (frequency_map >= threshold).astype(np.uint8) * 255
    return integrated_edge, frequency_map
def apply_colormap_grayscale_rgba(data, mask, vmin=0, vmax=255):
    data_norm = (data - vmin) / (vmax - vmin)
    data_norm = np.clip(data_norm, 0, 1)
    gray_values = ((1 - data_norm) * 255).astype(np.uint8)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray_values
    rgba[:, :, 1] = gray_values
    rgba[:, :, 2] = gray_values
    rgba[:, :, 3] = (mask * 255).astype(np.uint8)
    return rgba
def extract_he_edges_aligned(he_path, desi_path, params_path, output_dir,
                             methods=['sobel', 'laplacian', 'wavelet_sobel', 'canny',
                                      'entropy', 'flow_rate', 'contour'],
                             max_edge_pixel_ratio=0.1,
                             final_frequency_top_ratio=0.3,
                             erosion_size=1):
    os.makedirs(output_dir, exist_ok=True)
    M_final, he_aligned, desi_img = get_coarse_alignment_matrix(he_path, desi_path, params_path)
    if he_aligned.shape[2] == 4:
        he_gray = cv2.cvtColor(he_aligned[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        he_gray = cv2.cvtColor(he_aligned, cv2.COLOR_RGB2GRAY)
    valid_mask = get_valid_mask(he_aligned, erosion_size=erosion_size)
    edges_dict = extract_edges_multi_method(he_gray, valid_mask, methods, max_edge_pixel_ratio)
    integrated_edge, frequency_map = integrate_edges(edges_dict, final_frequency_top_ratio)
    mask_rgba = np.zeros((*valid_mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, :3] = 255
    mask_rgba[:, :, 3] = (valid_mask * 255).astype(np.uint8)
    tifffile.imwrite(f"{output_dir}/valid_mask.tiff", mask_rgba, compression=None)
    for method, edge in edges_dict.items():
        edge_mask = edge > 0
        edge_rgba = apply_colormap_grayscale_rgba(edge * 255, edge_mask, 0, 255)
        tifffile.imwrite(f"{output_dir}/edge_{method}.tiff", edge_rgba, compression=None)
    freq_mask = frequency_map > 0
    freq_rgba = apply_colormap_grayscale_rgba(frequency_map, freq_mask, 0, 1)
    tifffile.imwrite(f"{output_dir}/edge_frequency_map.tiff", freq_rgba, compression=None)
    integrated_mask = integrated_edge > 0
    integrated_rgba = apply_colormap_grayscale_rgba(integrated_edge, integrated_mask, 0, 255)
    tifffile.imwrite(f"{output_dir}/edge_integrated.tiff", integrated_rgba, compression=None)
    overlay = he_aligned.copy()
    edge_positions = integrated_edge > 0
    overlay[edge_positions, 0] = 0
    overlay[edge_positions, 1] = 0
    overlay[edge_positions, 2] = 0
    if overlay.shape[2] == 4:
        overlay[~valid_mask, 3] = 0
    tifffile.imwrite(f"{output_dir}/edge_overlay_on_he.tiff", overlay, compression=None)
    return M_final, he_aligned, integrated_edge, frequency_map, edges_dict, valid_mask
def load_alignment_params(params_path):
    params = {}
    with open(params_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Translation_X:' in line:
                params['tx'] = float(line.split(':')[1].strip())
            elif 'Translation_Y:' in line:
                params['ty'] = float(line.split(':')[1].strip())
            elif 'Rotation_Deg:' in line:
                params['rot'] = float(line.split(':')[1].strip())
    return params['tx'], params['ty'], params['rot']
def get_coarse_alignment_matrix(he_path, desi_path, params_path):
    tx, ty, rot = load_alignment_params(params_path)
    he_img = tifffile.imread(he_path)
    desi_img = tifffile.imread(desi_path)
    desi_mask = desi_img[:, :, 3] > 0 if desi_img.shape[2] == 4 else cv2.cvtColor(desi_img, cv2.COLOR_RGB2GRAY) < 240
    contours, _ = cv2.findContours(desi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (desi_cx, desi_cy), _ = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    rot_center = (desi_cx, desi_cy)
    M_T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    M_R_2x3 = cv2.getRotationMatrix2D(rot_center, rot, 1.0)
    M_R = np.vstack([M_R_2x3, [0, 0, 1]])
    M_final = M_R @ M_T
    canvas_h, canvas_w = desi_img.shape[:2]
    he_aligned = cv2.warpAffine(he_img, M_final[:2, :], (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255, 0))
    return M_final, he_aligned, desi_img
def process_and_append_grid_coordinates(input_csv_path, output_csv_path):
    def get_line_centers(indices):
        if len(indices) == 0:
            return []
        centers = []
        split_indices = np.where(np.diff(indices) > 1)[0] + 1
        groups = np.split(indices, split_indices)
        for group in groups:
            if len(group) > 0:
                centers.append(np.mean(group))
        return np.array(centers)
    def generate_robust_grid(centers, limit):
        if len(centers) < 2:
            return []
        diffs = np.diff(centers)
        min_spacing = 1
        valid_diffs = diffs[diffs >= min_spacing]
        if len(valid_diffs) == 0:
            return []
        spacing = np.median(valid_diffs)
        mid_idx = len(centers) // 2
        anchor = centers[mid_idx]
        grid_backward = np.arange(anchor, -spacing, -spacing)
        grid_forward = np.arange(anchor + spacing, limit + spacing, spacing)
        full_grid = np.concatenate((grid_backward, grid_forward))
        full_grid = np.sort(full_grid)
        valid_coords = [int(round(x)) for x in full_grid if 0 <= x <= limit]
        valid_coords = sorted(list(set(valid_coords)))
        return valid_coords
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found: {input_csv_path}")
        return
    print(f"Reading CSV file: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(input_csv_path, encoding='gbk')
    if 'original' not in df.columns:
        print("Error: Column 'original' not found in CSV.")
        return
    df['grid_x'] = np.empty((len(df), 0)).tolist()
    df['grid_y'] = np.empty((len(df), 0)).tolist()
    total_rows = len(df)
    print(f"Starting processing for {total_rows} images...")
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Generating Grids"):
        img_path = row['original']
        grid_x_list = []
        grid_y_list = []
        try:
            if not os.path.exists(str(img_path)):
                continue
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            abs_grad_x = cv2.normalize(np.absolute(grad_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            abs_grad_y = cv2.normalize(np.absolute(grad_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            projection_threshold = 10
            row_max = np.max(abs_grad_y, axis=1)
            raw_rows = np.where(row_max > projection_threshold)[0]
            col_max = np.max(abs_grad_x, axis=0)
            raw_cols = np.where(col_max > projection_threshold)[0]
            center_rows = get_line_centers(raw_rows)
            center_cols = get_line_centers(raw_cols)
            grid_y_list = generate_robust_grid(center_rows, h)
            grid_x_list = generate_robust_grid(center_cols, w)
        except Exception as e:
            pass
        df.at[index, 'grid_x'] = grid_x_list
        df.at[index, 'grid_y'] = grid_y_list
    print("\nFiltering results...")
    initial_count = len(df)
    df_filtered = df[df['grid_x'].apply(lambda x: len(x) > 1) & df['grid_y'].apply(lambda x: len(x) > 1)].copy()
    final_count = len(df_filtered)
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows due to detection failure.")
    if df_filtered.empty:
        print("❌ Error: No valid grids detected in any image.")
        return
    try:
        df_filtered.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ Success! Updated CSV saved to: {output_csv_path}")
        print(f"Total valid entries: {final_count}")
        print("\n--- Example Data (Row 0) ---")
        print(f"ID: {df_filtered.iloc[0].get('ID', 'N/A')}")
        print(f"Grid X (First 5): {df_filtered.iloc[0]['grid_x'][:5]}")
        print(f"Grid Y (First 5): {df_filtered.iloc[0]['grid_y'][:5]}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
def read_process_fast(path, target_size=800):
    img = tifffile.imread(path)
    h, w = img.shape[:2]
    scale = target_size / max(h, w) if target_size else 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if img_small.shape[2] == 4:
        mask = img_small[:, :, 3] > 0
    else:
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask = mask > 0
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No contour found in {path}")
    (cx, cy), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return {
        'img_small': img_small,
        'mask': mask,
        'center': (cx, cy),
        'scale': scale,
        'orig_shape': (h, w)
    }
def calculate_iou_fast(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    i_area = np.count_nonzero(intersection)
    u_area = np.count_nonzero(union)
    return i_area / u_area if u_area > 0 else 0
def optimize_rotation(he_mask_centered, desi_mask, rot_center):
    h, w = desi_mask.shape
    stages = [(30, 10), (10, 1), (1, 0.1), (0.1, 0.01)]
    current_center_angle = 0.0
    max_iou = 0.0
    for search_range, step in stages:
        start = current_center_angle - search_range
        end = current_center_angle + search_range + (step / 10.0)
        angles = (np.arange(start, end, step) + 180) % 360 - 180
        stage_best_iou = -1
        stage_best_angle = current_center_angle
        for angle in angles:
            M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
            rotated = cv2.warpAffine(he_mask_centered, M, (w, h), flags=cv2.INTER_NEAREST)
            iou = calculate_iou_fast(rotated, desi_mask)
            if iou > stage_best_iou:
                stage_best_iou = iou
                stage_best_angle = angle
        if stage_best_iou > max_iou:
            max_iou = stage_best_iou
        current_center_angle = stage_best_angle
    return current_center_angle, max_iou
def align_fast(he_path, desi_path, output_dir):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    calc_size = 800
    desi_data = read_process_fast(desi_path, target_size=calc_size)
    he_img_orig = tifffile.imread(he_path)
    h_he, w_he = he_img_orig.shape[:2]
    scale = desi_data['scale']
    he_small = cv2.resize(he_img_orig, (int(w_he * scale), int(h_he * scale)), interpolation=cv2.INTER_NEAREST)
    if he_small.shape[2] == 4:
        he_mask = he_small[:, :, 3] > 0
    else:
        gray = cv2.cvtColor(he_small, cv2.COLOR_RGB2GRAY)
        _, he_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        he_mask = he_mask > 0
    he_mask = he_mask.astype(np.uint8)
    c_contours, _ = cv2.findContours(he_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (he_cx, he_cy), _ = cv2.minEnclosingCircle(max(c_contours, key=cv2.contourArea))
    dx_small = desi_data['center'][0] - he_cx
    dy_small = desi_data['center'][1] - he_cy
    canvas_h, canvas_w = desi_data['mask'].shape
    M_trans = np.float32([[1, 0, dx_small], [0, 1, dy_small]])
    he_mask_centered = cv2.warpAffine(he_mask, M_trans, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST)
    rot_center = desi_data['center']
    best_angle, final_iou = optimize_rotation(he_mask_centered, desi_data['mask'], rot_center)
    dx_orig = dx_small / scale
    dy_orig = dy_small / scale
    final_angle = round(best_angle, 2)
    rot_center_orig = (rot_center[0] / scale, rot_center[1] / scale)
    preview_size = 1000
    d_scale = min(preview_size / canvas_h, preview_size / canvas_w)
    disp_w, disp_h = int(canvas_w * d_scale), int(canvas_h * d_scale)
    desi_disp = cv2.resize(desi_data['img_small'], (disp_w, disp_h))
    M_T = np.float32([[1, 0, dx_small], [0, 1, dy_small], [0, 0, 1]])
    M_R_2x3 = cv2.getRotationMatrix2D(rot_center, final_angle, 1.0)
    M_R = np.vstack([M_R_2x3, [0, 0, 1]])
    M_Final = M_R @ M_T
    he_aligned = cv2.warpAffine(he_small, M_Final[:2, :], (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    he_disp = cv2.resize(he_aligned, (disp_w, disp_h))
    composite = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)
    oy, ox = (preview_size - disp_h) // 2, (preview_size - disp_w) // 2
    dn, hn = desi_disp.astype(float) / 255.0, he_disp.astype(float) / 255.0
    roi = np.zeros((disp_h, disp_w, 3), dtype=float)
    roi[:, :, 1] = dn[:, :, :3].mean(axis=2) * (dn[:, :, 3] if dn.shape[2] == 4 else 1)
    h_val = hn[:, :, :3].mean(axis=2) * (hn[:, :, 3] if hn.shape[2] == 4 else 1)
    roi[:, :, 0] = h_val
    roi[:, :, 2] = h_val
    composite[oy:oy + disp_h, ox:ox + disp_w] = (np.clip(roi, 0, 1) * 255).astype(np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(composite)
    plt.axis('off')
    plt.title(f"Aligned: Rot {final_angle} deg, IoU {final_iou:.2%}")
    plt.savefig(os.path.join(output_dir, "result_preview.png"), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    with open(os.path.join(output_dir, "result_params.txt"), "w") as f:
        f.write(f"Target (DESI): {os.path.basename(desi_path)}\n")
        f.write(f"Source (HE):   {os.path.basename(he_path)}\n")
        f.write(f"IoU:           {final_iou:.4f}\n")
        f.write(f"Time:          {time.time() - start_time:.2f}s\n\n")
        f.write(f"Translation_X: {dx_orig:.2f}\n")
        f.write(f"Translation_Y: {dy_orig:.2f}\n")
        f.write(f"Rotation_Deg:  {final_angle:.2f}\n")
        f.write(f"Rotation_Center_X: {rot_center_orig[0]:.2f}\n")
        f.write(f"Rotation_Center_Y: {rot_center_orig[1]:.2f}\n")
        f.write("\nNote: Rotation center is DESI image center in original pixels.\n")
        f.write("Apply Translation first, then Rotation around the specified center.")
def read_image(path):
    pil_img = Image.open(path)
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    img = np.array(pil_img)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
def save_image(image, path):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    Image.fromarray(img_rgb).save(path, compression='tiff_deflate')
def read_first_params(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    tx = float(re.search(r'Translation_X:\s*([-\d.]+)', content).group(1))
    ty = float(re.search(r'Translation_Y:\s*([-\d.]+)', content).group(1))
    angle = float(re.search(r'Rotation_Deg:\s*([-\d.]+)', content).group(1))
    try:
        cx = float(re.search(r'Rotation_Center_X:\s*([-\d.]+)', content).group(1))
        cy = float(re.search(r'Rotation_Center_Y:\s*([-\d.]+)', content).group(1))
        return tx, ty, angle, (cx, cy)
    except:
        return tx, ty, angle, None
def read_second_params(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    tx = -float(re.search(r'Translation_X:\s*([-\d.]+)', content).group(1))
    ty = -float(re.search(r'Translation_Y:\s*([-\d.]+)', content).group(1))
    angle = float(re.search(r'Rotation_Deg:\s*([-\d.]+)', content).group(1))
    try:
        cx = float(re.search(r'Rotation_Center_X:\s*([-\d.]+)', content).group(1))
        cy = float(re.search(r'Rotation_Center_Y:\s*([-\d.]+)', content).group(1))
        return tx, ty, angle, (cx, cy)
    except:
        return tx, ty, angle, None
def apply_transform(image, tx, ty, angle, canvas_size, rot_center=None):
    h, w = canvas_size
    src_h, src_w = image.shape[:2]
    if (src_h, src_w) != (h, w):
        result = np.zeros((h, w, 4), dtype=image.dtype)
        result[:min(src_h, h), :min(src_w, w)] = image[:min(src_h, h), :min(src_w, w)]
        image = result
    if rot_center is None:
        rot_center = (w / 2, h / 2)
    M_T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    M_R_2x3 = cv2.getRotationMatrix2D(rot_center, angle, 1.0)
    M_R = np.vstack([M_R_2x3, [0, 0, 1]])
    M_Final = M_R @ M_T
    return cv2.warpAffine(image, M_Final[:2, :], (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))
def two_stage_alignment(he_path, params1_path, params2_path, target_path, output_path, save_stage1=True):
    he_image = read_image(he_path)
    canvas_size = read_image(target_path).shape[:2]
    tx1, ty1, angle1, rot_center1 = read_first_params(params1_path)
    stage1 = apply_transform(he_image, tx1, ty1, angle1, canvas_size, rot_center1)
    if save_stage1:
        save_image(stage1, output_path.replace('_final', '_stage1'))
    tx2, ty2, angle2, rot_center2 = read_second_params(params2_path)
    stage2 = apply_transform(stage1, tx2, ty2, angle2, canvas_size, rot_center2)
    save_image(stage2, output_path)
    return stage2
def process_msi_file2(input_file: str,
                      target_mz: Optional[Union[float, str]] = None,  
                      n_clusters: int = 7,  
                      target_dpi: int = 1000,
                      sampling_interval: float = 2.0,
                      colormap: str = 'viridis',
                      msi_resolution: float = 50.0):
    file_path = Path(input_file)
    if target_mz is not None:
        folder_suffix = f"_{target_mz}" if isinstance(target_mz, str) else f"_mz_{target_mz:.4f}"
        output_dir = file_path.parent / (file_path.stem + folder_suffix)
    else:
        output_dir = file_path.parent / file_path.stem
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Loading data: {file_path.name} ...")
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    data_start_col = 22
    pattern = re.compile(r'-(\d+)-(\d+)$')
    coords, valid_indices = [], []
    for idx, name in enumerate(df.columns[data_start_col:]):
        match = pattern.search(str(name))
        if match:
            coords.append((int(match.group(1)), int(match.group(2))))
            valid_indices.append(idx)
    xs, ys = zip(*coords)
    width, height = max(xs) + 1, max(ys) + 1
    
    
    mz_values = df.iloc[:, 0].values
    pixel_data = df.iloc[:, [x + data_start_col for x in valid_indices]].values
    cmap = plt.get_cmap(colormap)
    
    
    
    
    tasks = []
    if target_mz is None:
        
        for i in range(len(mz_values)):
            tasks.append((mz_values[i], pixel_data[i, :]))
    elif isinstance(target_mz, (int, float)):
        
        
        diff = np.abs(mz_values - target_mz)
        candidates_mask = diff <= 0.01
        if not np.any(candidates_mask):
            print(f"❌ Error: No m/z found close to {target_mz} (+/- 0.01 Da)")
            return
        
        candidates_indices = np.where(candidates_mask)[0]
        best_idx = candidates_indices[np.argmin(diff[candidates_indices])]
        found_mz = mz_values[best_idx]
        print(f"✓ Target {target_mz} -> Found closest m/z: {found_mz:.4f} (diff: {diff[best_idx]:.5f})")
        tasks.append((found_mz, pixel_data[best_idx, :]))
    elif target_mz in ['kmeans', 'kmeans_exp']:
        
        print(f"✓ Calculating KMeans (n={n_clusters}, mode={target_mz})...")
        
        pixel_features = pixel_data.T
        pixel_features = np.nan_to_num(pixel_features.astype(np.float32))
        
        valid_pixel_mask = np.any(pixel_features > 0, axis=1)
        valid_pixel_indices = np.where(valid_pixel_mask)[0]
        if len(valid_pixel_indices) < n_clusters:
            print("❌ Error: Not enough valid pixels for clustering.")
            return
        
        valid_features = pixel_features[valid_pixel_indices]
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, batch_size=2048)
        labels = kmeans.fit_predict(valid_features)
        
        result_flat = np.zeros(pixel_data.shape[1], dtype=np.float32)
        if target_mz == 'kmeans':
            
            
            result_flat[valid_pixel_indices] = labels.astype(np.float32) + 1.0
        elif target_mz == 'kmeans_exp':
            
            norms = np.linalg.norm(valid_features, axis=1)
            
            final_vals = np.zeros_like(labels, dtype=np.float32)
            for cid in range(n_clusters):
                mask_c = (labels == cid)
                if not np.any(mask_c): continue
                c_norms = norms[mask_c]
                c_min, c_max = c_norms.min(), c_norms.max()
                if c_max > c_min:
                    norm_relative = (c_norms - c_min) / (c_max - c_min)
                else:
                    norm_relative = 0.5
                
                final_vals[mask_c] = (cid + 1.0) + (norm_relative * 0.9)
            result_flat[valid_pixel_indices] = final_vals
        tasks.append((target_mz, result_flat))
    else:
        print(f"❌ Error: Invalid target_mz parameter: {target_mz}")
        return
    
    
    
    for identifier, flat_data in tqdm(tasks, desc="Processing", unit="img"):
        
        matrix = np.zeros((height, width), dtype=np.float32)
        
        matrix[list(ys), list(xs)] = np.nan_to_num(flat_data)
        
        if isinstance(identifier, (float, int)):
            fname_id = f"{identifier:.4f}"
        else:
            fname_id = str(identifier)
        
        for is_sampling in [False, True]:
            if is_sampling:
                zoom_factor = 1.0 / sampling_interval
                proc_matrix = zoom(matrix, zoom_factor, order=1)  
                suffix = "sampling"
                scale = sampling_interval
            else:
                proc_matrix = matrix.copy()
                suffix = "original"
                scale = 1.0
            
            valid_mask = proc_matrix > 0
            if valid_mask.any():
                v_min, v_max = proc_matrix[valid_mask].min(), proc_matrix[valid_mask].max()
                if v_max > v_min:
                    proc_matrix[valid_mask] = (proc_matrix[valid_mask] - v_min) / (v_max - v_min)
                else:
                    proc_matrix[valid_mask] = 1.0
            
            h, w = proc_matrix.shape
            phys_w_inch = (w * msi_resolution * scale) / 25400.0
            target_w = int(round(phys_w_inch * target_dpi))
            if target_w <= 0: target_w = 1  
            target_h = int(round(h * (target_w / w)))
            if target_h <= 0: target_h = 1
            
            
            rgba = (cmap(proc_matrix) * 255).astype(np.uint8)
            rgba[:, :, 3] = (proc_matrix > 0).astype(np.uint8) * 255
            
            img = Image.fromarray(rgba, mode='RGBA')
            img_resized = img.resize((target_w, target_h), resample=Image.NEAREST)
            output_path = output_dir / f"{fname_id}_{suffix}.tiff"
            img_resized.save(str(output_path), format='TIFF',
                             compression='tiff_deflate', dpi=(target_dpi, target_dpi))
def calculate_sampling_interval2(original_resolution: float, target_resolution: float) -> float:
    if target_resolution <= 0 or original_resolution <= 0:
        raise ValueError("Resolution must be greater than 0")
    if target_resolution > original_resolution:
        raise ValueError("Target resolution cannot be greater than original resolution")
    return round(original_resolution / target_resolution, 2)
def process_msi_file3(input_file: str,
                      target_mz: Optional[Union[float, str]] = None,
                      n_clusters: int = 7,
                      target_dpi: int = 1000,
                      sampling_interval: float = 2.0,
                      colormap: str = 'viridis',
                      msi_resolution: float = 50.0):
    file_path = Path(input_file)
    if target_mz is not None:
        folder_suffix = f"_{target_mz}" if isinstance(target_mz, str) else f"_mz_{target_mz:.4f}"
        output_dir = file_path.parent / (file_path.stem + folder_suffix)
    else:
        output_dir = file_path.parent / file_path.stem
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Loading data: {file_path.name} ...")
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    data_start_col = 22
    pattern = re.compile(r'-(\d+)-(\d+)$')
    coords, valid_indices = [], []
    for idx, name in enumerate(df.columns[data_start_col:]):
        match = pattern.search(str(name))
        if match:
            coords.append((int(match.group(1)), int(match.group(2))))
            valid_indices.append(idx)
    xs, ys = zip(*coords)
    xs, ys = list(xs), list(ys)
    width, height = max(xs) + 1, max(ys) + 1
    mz_values = df.iloc[:, 0].values
    pixel_data = df.iloc[:, [x + data_start_col for x in valid_indices]].values
    cmap = plt.get_cmap(colormap)
    tasks = []
    if target_mz is None:
        for i in range(len(mz_values)):
            tasks.append({
                "identifier": mz_values[i],
                "flat_data": pixel_data[i, :],
                "grid_type": "sparse",
                "width": width,
                "height": height,
                "xs": xs,
                "ys": ys
            })
    elif isinstance(target_mz, (int, float)):
        diff = np.abs(mz_values - target_mz)
        candidates_mask = diff <= 0.01
        if not np.any(candidates_mask):
            print(f"❌ Error: No m/z found close to {target_mz} (+/- 0.01 Da)")
            return
        candidates_indices = np.where(candidates_mask)[0]
        best_idx = candidates_indices[np.argmin(diff[candidates_indices])]
        found_mz = mz_values[best_idx]
        print(f"✓ Target {target_mz} -> Found closest m/z: {found_mz:.4f} (diff: {diff[best_idx]:.5f})")
        tasks.append({
            "identifier": found_mz,
            "flat_data": pixel_data[best_idx, :],
            "grid_type": "sparse",
            "width": width,
            "height": height,
            "xs": xs,
            "ys": ys
        })
    elif target_mz in ['kmeans', 'kmeans_exp']:
        print(f"✓ Calculating KMeans after sampling (n={n_clusters}, mode={target_mz})...")
        zoom_factor = 1.0 / sampling_interval
        sampled_spectra = []
        sampled_shape = None
        for i in range(pixel_data.shape[0]):
            mz_matrix = np.zeros((height, width), dtype=np.float32)
            mz_matrix[ys, xs] = np.nan_to_num(pixel_data[i, :]).astype(np.float32)
            sampled_matrix = zoom(mz_matrix, zoom_factor, order=1)
            if sampled_shape is None:
                sampled_shape = sampled_matrix.shape
            sampled_spectra.append(sampled_matrix.reshape(-1))
        sampled_pixel_data = np.stack(sampled_spectra, axis=0)
        pixel_features = sampled_pixel_data.T
        pixel_features = np.nan_to_num(pixel_features.astype(np.float32))
        valid_pixel_mask = np.any(pixel_features > 0, axis=1)
        valid_pixel_indices = np.where(valid_pixel_mask)[0]
        if len(valid_pixel_indices) < n_clusters:
            print("❌ Error: Not enough valid pixels for clustering after sampling.")
            return
        valid_features = pixel_features[valid_pixel_indices]
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            batch_size=2048
        )
        labels = kmeans.fit_predict(valid_features)
        result_flat = np.zeros(pixel_features.shape[0], dtype=np.float32)
        if target_mz == 'kmeans':
            result_flat[valid_pixel_indices] = labels.astype(np.float32) + 1.0
        elif target_mz == 'kmeans_exp':
            norms = np.linalg.norm(valid_features, axis=1)
            final_vals = np.zeros_like(labels, dtype=np.float32)
            for cid in range(n_clusters):
                mask_c = (labels == cid)
                if not np.any(mask_c):
                    continue
                c_norms = norms[mask_c]
                c_min, c_max = c_norms.min(), c_norms.max()
                if c_max > c_min:
                    norm_relative = (c_norms - c_min) / (c_max - c_min)
                else:
                    norm_relative = np.full(c_norms.shape, 0.5, dtype=np.float32)
                final_vals[mask_c] = (cid + 1.0) + (norm_relative * 0.9)
            result_flat[valid_pixel_indices] = final_vals
        sampled_h, sampled_w = sampled_shape
        tasks.append({
            "identifier": target_mz,
            "flat_data": result_flat,
            "grid_type": "dense_sampled_kmeans",
            "sampled_width": sampled_w,
            "sampled_height": sampled_h,
            "orig_width": width,
            "orig_height": height
        })
    else:
        print(f"❌ Error: Invalid target_mz parameter: {target_mz}")
        return
    for task in tqdm(tasks, desc="Processing", unit="img"):
        identifier = task["identifier"]
        if isinstance(identifier, (float, int)):
            fname_id = f"{identifier:.4f}"
        else:
            fname_id = str(identifier)
        if task["grid_type"] == "sparse":
            matrix = np.zeros((task["height"], task["width"]), dtype=np.float32)
            matrix[task["ys"], task["xs"]] = np.nan_to_num(task["flat_data"])
            render_plan = [
                ("original", matrix.copy(), 1.0),
                ("sampling", zoom(matrix, 1.0 / sampling_interval, order=1), sampling_interval)
            ]
        elif task["grid_type"] == "dense_sampled_kmeans":
            sampled_matrix = np.nan_to_num(task["flat_data"]).reshape(
                task["sampled_height"], task["sampled_width"]
            ).astype(np.float32)
            zoom_back_h = task["orig_height"] / task["sampled_height"]
            zoom_back_w = task["orig_width"] / task["sampled_width"]
            original_matrix = zoom(sampled_matrix, (zoom_back_h, zoom_back_w), order=0)
            original_matrix = original_matrix[:task["orig_height"], :task["orig_width"]]
            if original_matrix.shape[0] < task["orig_height"] or original_matrix.shape[1] < task["orig_width"]:
                padded = np.zeros((task["orig_height"], task["orig_width"]), dtype=np.float32)
                padded[:original_matrix.shape[0], :original_matrix.shape[1]] = original_matrix
                original_matrix = padded
            render_plan = [
                ("original", original_matrix, 1.0),
                ("sampling", sampled_matrix.copy(), sampling_interval)
            ]
        else:
            print(f"❌ Error: Unknown grid_type: {task['grid_type']}")
            return
        for suffix, proc_matrix, scale in render_plan:
            valid_mask = proc_matrix > 0
            if valid_mask.any():
                v_min, v_max = proc_matrix[valid_mask].min(), proc_matrix[valid_mask].max()
                if v_max > v_min:
                    proc_matrix[valid_mask] = (proc_matrix[valid_mask] - v_min) / (v_max - v_min)
                else:
                    proc_matrix[valid_mask] = 1.0
            h, w = proc_matrix.shape
            phys_w_inch = (w * msi_resolution * scale) / 25400.0
            target_w = int(round(phys_w_inch * target_dpi))
            if target_w <= 0:
                target_w = 1
            target_h = int(round(h * (target_w / w)))
            if target_h <= 0:
                target_h = 1
            rgba = (cmap(proc_matrix) * 255).astype(np.uint8)
            rgba[:, :, 3] = (proc_matrix > 0).astype(np.uint8) * 255
            img = Image.fromarray(rgba, mode='RGBA')
            img_resized = img.resize((target_w, target_h), resample=Image.NEAREST)
            output_path = output_dir / f"{fname_id}_{suffix}.tiff"
            img_resized.save(
                str(output_path),
                format='TIFF',
                compression='tiff_deflate',
                dpi=(target_dpi, target_dpi)
            )
