from supporting_function1 import *
import re
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import binary_dilation
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
############################################################
#                                                          #
#             You only need to modify here                 #
#                                                          #
############################################################
INPUT_TIFF_PATH = str(Path(__file__).resolve().parent.parent / "demo for super-resolution and sparse-sampling" / "HE")
MODEL_WEIGHTS = str(Path(__file__).resolve().parent.parent / "weights of SRP" / "super-resolution.pth")























downsampling_dpi = 2000
PATCH_SIZE = 48
MSI_RESOLUTION = 0.4
HE_PIXEL_SIZE = 0.5
tiff_folder = Path(INPUT_TIFF_PATH)
tiff_files = list(tiff_folder.glob("*.tiff"))
for i, tiff_file in enumerate(tiff_files, 1):
    print(f"Processing [{i}/{len(tiff_files)}]: {tiff_file.name}")
    try:
        start_time = time.time()
        INPUT_HE = str(tiff_file)
        INPUT_XLS = INPUT_HE.replace('HE', 'SRP')
        INPUT_XLS = INPUT_XLS.replace('.tiff', '.xls')
        process_msi_file(
            input_file=INPUT_XLS,
            output_first_row_tiff=True,
            output_gradient_tiff=True,
            gradient_methods=['wavelet_sobel'],
            target_dpi=downsampling_dpi,
            first_row_colormap='rainbow',
            max_edge_pixel_ratio=0.50,
            final_frequency_top_ratio=1
        )
        align_fast(he_path=INPUT_HE,
                   desi_path=INPUT_XLS.replace('.xls', '_first_mz.tiff'),
                   output_dir=INPUT_XLS.replace('.xls', 'Alignment_Result'))
        he_path = INPUT_HE
        desi_path = INPUT_XLS.replace('.xls', '_first_mz.tiff')
        params_path = INPUT_XLS.replace('.xls', 'Alignment_Result/result_params.txt')
        output_dir = INPUT_XLS.replace('.xls', 'HE_Edges')
        M_final, he_aligned, integrated_edge, frequency_map, edges_dict, valid_mask = extract_he_edges_aligned(
            he_path, desi_path, params_path, output_dir,
            methods=['wavelet_sobel'],
            max_edge_pixel_ratio=0.5,
            final_frequency_top_ratio=0.05
        )
        path_msi = INPUT_XLS.replace('.xls', '_gradient_wavelet_sobel_density.tiff')
        path_he = INPUT_XLS.replace('.xls', 'HE_Edges/edge_wavelet_sobel.tiff')
        output_dir = INPUT_XLS.replace('.xls', 'Alignment_Result2')
        aligner = AdaptiveGradientAlignment(output_dir)
        global_params = aligner.process(
            path_a=path_msi,
            path_b=path_he,
            patch_size=5000000,
            max_shift=0,
            max_angle=0
        )
        two_stage_alignment(
            he_path=INPUT_HE,
            params1_path=INPUT_XLS.replace('.xls', 'Alignment_Result/result_params.txt'),
            params2_path=INPUT_XLS.replace('.xls', 'Alignment_Result2/alignment_params.txt'),
            target_path=INPUT_XLS.replace('.xls', '_first_mz.tiff'),
            output_path=INPUT_XLS.replace('.xls', 'HE_Edges/HE_aligned_final.tiff')
        )
        img1 = Image.open(INPUT_XLS.replace('.xls', 'HE_Edges/HE_aligned_final.tiff')).convert('RGBA')
        img2 = Image.open(INPUT_XLS.replace('.xls', '_first_mz.tiff')).convert('RGBA')
        img1.paste(img2, (0, 0), img2)
        img1.save(INPUT_XLS.replace('.xls', 'HE_Edges/Aligned_Checked.tiff'))
        process_msi_file_LR(input_file=INPUT_XLS, target_dpi=downsampling_dpi,
                            target_mz=None,
                            sampling_interval=calculate_sampling_interval2(200, 50),
                            colormap='viridis')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"❌ ERROR: {tiff_file.name} - {e}")
HE_FOLDER_PATH = INPUT_TIFF_PATH
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def process_images(b=256, target_msi_res=100.0, he_pixel_res=0.5, use_sliding_window=True, stride=0.75,
                   gene_name='AAMP'):
    he_folder = Path(HE_FOLDER_PATH)
    tiff_files = list(he_folder.glob('*.tiff')) + list(he_folder.glob('*.tif'))
    print(f"Found {len(tiff_files)} TIFF files.")
    for tiff_file in tiff_files:
        base_name = tiff_file.stem
        xls_base_path = Path(str(tiff_file.parent).replace('HE', 'SRP'))
        xls_subfolder = xls_base_path / f"{base_name}"
        print(f"\n{'=' * 50}")
        print(f"Processing File: {base_name}")
        original_file = xls_subfolder / f"1_original.tiff"
        sampling_file = xls_subfolder / f"1_sampling.tiff"
        he_edges_folder = xls_base_path / f"{base_name}HE_Edges"
        he_aligned_file = he_edges_folder / "HE_aligned_final.tiff"
        output_base = he_folder
        output_original = output_base / 'original'
        output_sampling = output_base / 'sampling'
        output_he_vis = output_base / 'HE_Heatmap'
        output_he_vis_x2 = output_base / 'HE_HeatmapX2'
        output_he_feat = output_base / 'HE_Feature'
        output_he_feat_x2 = output_base / 'HE_FeatureX2'
        output_original.mkdir(parents=True, exist_ok=True)
        output_sampling.mkdir(parents=True, exist_ok=True)
        output_he_vis.mkdir(parents=True, exist_ok=True)
        output_he_vis_x2.mkdir(parents=True, exist_ok=True)
        output_he_feat.mkdir(parents=True, exist_ok=True)
        output_he_feat_x2.mkdir(parents=True, exist_ok=True)
        downsample_ratio = None
        if original_file.exists() and he_aligned_file.exists():
            try:
                with Image.open(original_file) as img_orig, Image.open(he_aligned_file) as img_he:
                    w_orig, h_orig = img_orig.size
                    w_he, h_he = img_he.size
                    downsample_ratio = (w_he / w_orig + h_he / h_orig) / 2
                    print(f"  Ratio (Image Calc): {downsample_ratio:.4f}")
            except:
                pass
        if downsample_ratio is None:
            downsample_ratio = target_msi_res / he_pixel_res
            print(f"  Ratio (Param Calc): {downsample_ratio:.4f}")
        if original_file.exists():
            cut_image_to_patches(original_file, output_original, base_name, b, use_sliding_window=use_sliding_window,
                                 stride=stride)
        if sampling_file.exists():
            cut_image_to_patches(sampling_file, output_sampling, base_name, b, use_sliding_window=use_sliding_window,
                                 stride=stride)
        if he_aligned_file.exists():
            print(f"  Generating HE Gradient Heatmap...")
            process_he_via_gradient_and_cut(
                image_path=he_aligned_file,
                output_vis_folder=output_he_vis,
                output_vis_folder_x2=output_he_vis_x2,
                output_feat_folder=output_he_feat,
                output_feat_folder_x2=output_he_feat_x2,
                folder_name=base_name,
                patch_size=b,
                downsample_ratio=downsample_ratio,
                use_sliding_window=use_sliding_window, stride=stride
            )
        else:
            print(f"  Warning: {he_aligned_file} not found.")
from torchvision import transforms
def process_he_via_gradient_and_cut(image_path, output_vis_folder, output_vis_folder_x2,
                                    output_feat_folder, output_feat_folder_x2, folder_name, patch_size,
                                    downsample_ratio, use_sliding_window=False, stride=None):
    try:
        img = Image.open(image_path).convert('RGB')
        W, H = img.size
        target_W = int(W / downsample_ratio)
        target_H = int(H / downsample_ratio)
        print(f"    Original HE: {W}x{H} -> Target Size: {target_W}x{target_H} (Ratio: {downsample_ratio:.2f})")
        print("    [1/3] Calculating Gradients & Density Heatmap...")
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        calc_device = DEVICE
        if W * H > 10000 * 10000:
            calc_device = torch.device('cpu')
            print("Image too large, switching to CPU for gradient calculation...")
        img_tensor = img_tensor.to(calc_device)
        gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=calc_device).view(1, 1,
                                                                                                                   3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=calc_device).view(1, 1,
                                                                                                                   3, 3)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        blurrer = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
        heatmap_density = blurrer(magnitude)
        min_val = heatmap_density.min()
        max_val = heatmap_density.max()
        heatmap_density = (heatmap_density - min_val) / (max_val - min_val + 1e-8)
        print("    [2/3] Downsampling to MSI resolution...")
        heatmap_tensor = F.interpolate(heatmap_density, size=(target_H, target_W), mode='bilinear',
                                       align_corners=False).squeeze()
        heatmap_np = heatmap_tensor.cpu().numpy()
        p5 = np.percentile(heatmap_np, 5)
        p99 = np.percentile(heatmap_np, 99)
        heatmap_norm = np.clip((heatmap_np - p5) / (p99 - p5 + 1e-8), 0, 1)
        heatmap_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).convert('RGB')
        print("    [3/3] Cutting patches...")
        if stride is None:
            stride = patch_size
        if not use_sliding_window:
            padded_W = ((target_W + patch_size - 1) // patch_size) * patch_size
            padded_H = ((target_H + patch_size - 1) // patch_size) * patch_size
        else:
            if target_W <= patch_size:
                padded_W = patch_size
            else:
                last_start_w = ((target_W - patch_size + stride - 1) // stride) * stride
                padded_W = last_start_w + patch_size
            if target_H <= patch_size:
                padded_H = patch_size
            else:
                last_start_h = ((target_H - patch_size + stride - 1) // stride) * stride
                padded_H = last_start_h + patch_size
        def pad_image(pil_img):
            new_img = Image.new('RGB', (padded_W, padded_H), (0, 0, 0))
            new_img.paste(pil_img, (0, 0))
            return new_img
        final_heatmap = pad_image(heatmap_img)
        if not use_sliding_window:
            xs = [x * patch_size for x in range(padded_W // patch_size)]
            ys = [y * patch_size for y in range(padded_H // patch_size)]
        else:
            xs = list(range(0, padded_W - patch_size + 1, stride))
            ys = list(range(0, padded_H - patch_size + 1, stride))
        count = 0
        configs = [
            (final_heatmap, output_vis_folder, output_feat_folder),
            (final_heatmap, output_vis_folder_x2, output_feat_folder_x2)
        ]
        for y_idx, m_top in enumerate(ys):
            for x_idx, m_left in enumerate(xs):
                m_right, m_bottom = m_left + patch_size, m_top + patch_size
                check_crop = final_heatmap.crop((m_left, m_top, m_right, m_bottom))
                if np.array(check_crop).mean() < 0:
                    continue
                for (h_pil, vis_dir, feat_dir) in configs:
                    vis_crop = h_pil.crop((m_left, m_top, m_right, m_bottom))
                    vis_name = f"HE_Heatmap_{folder_name}_x{x_idx:04d}_y{y_idx:04d}.tiff"
                    vis_crop.save(vis_dir / vis_name)
                    feat_name = f"HE_Feature_{folder_name}_x{x_idx:04d}_y{y_idx:04d}.npy"
                    open(feat_dir / feat_name, 'w').close()
                count += 1
        print(f"    Successfully saved {count} patches (Gradient Density Heatmaps).")
        img.close()
    except Exception as e:
        print(f"    Error in processing HE gradient: {e}")
        import traceback
        traceback.print_exc()
def cut_image_to_patches(image_path, output_folder, folder_name, patch_size, use_sliding_window=False, stride=None):
    try:
        img = Image.open(image_path)
        original_mode = img.mode
        if original_mode != 'RGBA':
            img = img.convert('RGBA')
        width, height = img.size
        original_name = image_path.stem
        if stride is None:
            stride = patch_size
        if not use_sliding_window:
            new_width = ((width + patch_size - 1) // patch_size) * patch_size
            new_height = ((height + patch_size - 1) // patch_size) * patch_size
        else:
            if width <= patch_size:
                new_width = patch_size
            else:
                last_start_w = ((width - patch_size + stride - 1) // stride) * stride
                new_width = last_start_w + patch_size
            if height <= patch_size:
                new_height = patch_size
            else:
                last_start_h = ((height - patch_size + stride - 1) // stride) * stride
                new_height = last_start_h + patch_size
        if new_width != width or new_height != height:
            padded_img = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            padded_img.paste(img, (0, 0))
            img.close()
            img = padded_img
            print(f"    Padded {original_name}: {width}x{height} -> {new_width}x{new_height}")
        if not use_sliding_window:
            xs = [x * patch_size for x in range(new_width // patch_size)]
            ys = [y * patch_size for y in range(new_height // patch_size)]
        else:
            xs = list(range(0, new_width - patch_size + 1, stride))
            ys = list(range(0, new_height - patch_size + 1, stride))
        for y_idx, upper in enumerate(ys):
            for x_idx, left in enumerate(xs):
                right = left + patch_size
                lower = upper + patch_size
                patch = img.crop((left, upper, right, lower))
                if np.array(patch)[..., 3].max() == 0:
                    continue
                if original_mode != 'RGBA':
                    patch = patch.convert(original_mode)
                output_name = f"{original_name}_{folder_name}_x{x_idx:04d}_y{y_idx:04d}.tiff"
                patch.save(output_folder / output_name)
        img.close()
    except Exception as e:
        print(f"Error cutting {image_path}: {e}")
process_images(b=PATCH_SIZE, target_msi_res=MSI_RESOLUTION, he_pixel_res=HE_PIXEL_SIZE,
               use_sliding_window=True, stride=24)
ROOT_DIR = INPUT_TIFF_PATH
OUTPUT_FILE = os.path.join(ROOT_DIR, "file_mapping.csv")
Final_output = os.path.join(ROOT_DIR, "file_mapping_with_grid.csv")
generate_unified_mapping_csv(ROOT_DIR, OUTPUT_FILE)
process_and_append_grid_coordinates(OUTPUT_FILE, Final_output)
def get_viridis_decoders():
    cmap = cm.get_cmap('viridis')
    lut_rgb = cmap(np.linspace(0, 1, 256))[:, :3].astype(np.float32)
    lut_sq_norm = np.sum(lut_rgb ** 2, axis=1)
    lut_luma = 0.299 * lut_rgb[:, 0] + 0.587 * lut_rgb[:, 1] + 0.114 * lut_rgb[:, 2]
    lut_luma_u8 = (lut_luma * 255).astype(np.int32)
    gray_to_scalar = np.zeros(256, dtype=np.uint8)
    for g in range(256):
        idx = np.argmin(np.abs(lut_luma_u8 - g))
        gray_to_scalar[g] = idx
    return lut_rgb, lut_sq_norm, gray_to_scalar
def stitch_images(file_list, tile_width, tile_height,
                  min_x_idx, min_y_idx, max_x_idx, max_y_idx,
                  stride_x_px=None, stride_y_px=None):
    lut_rgb, lut_sq_norm, gray_to_scalar = get_viridis_decoders()
    if stride_x_px is None:
        stride_x_px = tile_width
    if stride_y_px is None:
        stride_y_px = tile_height
    stride_x_px = int(stride_x_px)
    stride_y_px = int(stride_y_px)
    n_tiles_x = (max_x_idx - min_x_idx + 1)
    n_tiles_y = (max_y_idx - min_y_idx + 1)
    canvas_width = (n_tiles_x - 1) * stride_x_px + tile_width
    canvas_height = (n_tiles_y - 1) * stride_y_px + tile_height
    acc_sum = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    acc_w = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    def _decode_viridis_to_scalar(img):
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3 and img.shape[2] == 3:
            if img.dtype == np.uint8:
                pix = img.astype(np.float32) / 255.0
            else:
                pix = img.astype(np.float32)
                if pix.max() > 1.0:
                    pix /= 255.0
            H, W, _ = pix.shape
            flat_pix = pix.reshape(-1, 3)
            dots = flat_pix @ lut_rgb.T
            metric = dots - 0.5 * lut_sq_norm
            indices = np.argmax(metric, axis=1)
            return indices.reshape(H, W).astype(np.uint8)
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            if info.max > 255:
                img = (img.astype(np.float32) / info.max * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        else:
            img_f = img.astype(np.float32)
            if img_f.max() <= 1.0:
                img_f *= 255.0
            img = np.clip(np.round(img_f), 0, 255).astype(np.uint8)
        return gray_to_scalar[img]
    for item in tqdm(file_list, desc="Stitching"):
        img = tifffile.imread(item['path'])
        img_decoded = _decode_viridis_to_scalar(img).astype(np.float32)
        pixel_x = (item['x_idx'] - min_x_idx) * stride_x_px
        pixel_y = (item['y_idx'] - min_y_idx) * stride_y_px
        h, w = img_decoded.shape
        y1 = min(pixel_y + h, canvas_height)
        x1 = min(pixel_x + w, canvas_width)
        region = img_decoded[:(y1 - pixel_y), :(x1 - pixel_x)]
        acc_sum[pixel_y:y1, pixel_x:x1] += region
        acc_w[pixel_y:y1, pixel_x:x1] += 1.0
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    valid = acc_w > 0
    canvas[valid] = np.clip(acc_sum[valid] / acc_w[valid], 0, 255).astype(np.uint8)
    return canvas
def apply_grid_postprocess(img, h_lines, v_lines, method="none"):
    method = (method or "none").lower()
    if method == "none":
        return img
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    H, W = img.shape
    output = img.copy()
    ys = sorted(list(set([0] + list(h_lines) + [H])))
    xs = sorted(list(set([0] + list(v_lines) + [W])))
    def stat_uint8(flat):
        if flat.size == 0:
            return 0
        if method == "mode":
            counts = np.bincount(flat, minlength=256)
            return int(np.argmax(counts))
        elif method == "mean":
            return int(np.clip(np.round(flat.mean()), 0, 255))
        elif method == "q25":
            return int(np.clip(np.round(np.quantile(flat, 0.25)), 0, 255))
        elif method == "q75":
            return int(np.clip(np.round(np.quantile(flat, 0.75)), 0, 255))
        else:
            raise ValueError(f"Unknown grid postprocess method: {method}")
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            if (y1 - y0) < 2 or (x1 - x0) < 2:
                continue
            ymid = (y0 + y1) // 2
            xmid = (x0 + x1) // 2
            if ymid <= y0: ymid = y0 + 1
            if ymid >= y1: ymid = y1 - 1
            if xmid <= x0: xmid = x0 + 1
            if xmid >= x1: xmid = x1 - 1
            sub_boxes = [
                (y0, ymid, x0, xmid),
                (y0, ymid, xmid, x1),
                (ymid, y1, x0, xmid),
                (ymid, y1, xmid, x1),
            ]
            for (sy0, sy1, sx0, sx1) in sub_boxes:
                block = img[sy0:sy1, sx0:sx1]
                if block.size == 0:
                    continue
                val = stat_uint8(block.ravel())
                output[sy0:sy1, sx0:sx1] = val
    return output
def parse_coords_from_id(id_str):
    match_x = re.search(r'[xX]_?(\d+)', id_str)
    match_y = re.search(r'[yY]_?(\d+)', id_str)
    if match_x and match_y:
        return int(match_x.group(1)), int(match_y.group(1))
    return None, None
def load_tiff_image(path, force_gray=False):
    try:
        path = str(path)
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            return torch.zeros(1 if force_gray else 3, 256, 256)
        img = tifffile.imread(path).astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = None
        if img.ndim == 3:
            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                img = img[:, :, :3]
            elif img.shape[0] == 4 and img.shape[1] > 4 and img.shape[2] > 4:
                alpha = img[3, :, :]
                img = img[:3, :, :]
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
        if vals.size == 0:
            raw_max = None
        else:
            raw_max = float(np.max(vals))
        if raw_max is not None and raw_max > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3 and img.shape[2] < img.shape[0]:
            img = img.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(np.clip(img, 0.0, 1.0)))
    except Exception as e:
        print(f"[Error] Loading {path}: {e}")
        return torch.zeros(1 if force_gray else 3, 256, 256)
def compute_boxes_manual(H, W, h_lines, v_lines):
    ys = sorted(list(set([0] + h_lines + [H])))
    xs = sorted(list(set([0] + v_lines + [W])))
    boxes = []
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            y0, y1 = ys[r], ys[r + 1]
            x0, x1 = xs[c], xs[c + 1]
            if (y1 - y0) < 2 or (x1 - x0) < 2:
                continue
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)
def calculate_metrics(img1, img2, mask=None):
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        if mask is not None:
            mask = mask[:min_h, :min_w]
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
    if mask is not None:
        img1_2d = img1.copy()
        img2_2d = img2.copy()
        img1_2d[~mask] = 0
        img2_2d[~mask] = 0
        img1_1d = img1[mask].astype(np.float32)
        img2_1d = img2[mask].astype(np.float32)
    else:
        img1_2d = img1
        img2_2d = img2
        img1_1d = img1.flatten().astype(np.float32)
        img2_1d = img2.flatten().astype(np.float32)
    metrics = {}
    try:
        if mask is not None:
            mask_psnr = mask & (img1 > 0)
            if mask_psnr.sum() < 10:
                psnr_val = np.nan
            else:
                diff = img1[mask_psnr].astype(np.float32) - img2[mask_psnr].astype(np.float32)
                mse_val = float(np.mean(diff ** 2))
                eps = 1e-8
                mse_val = max(mse_val, eps)
                psnr_val = 10.0 * np.log10((255.0 ** 2) / mse_val)
                if not np.isfinite(psnr_val):
                    psnr_val = 100.0
                else:
                    psnr_val = min(psnr_val, 100.0)
        else:
            psnr_val = psnr(img1_2d, img2_2d, data_range=255)
        metrics['PSNR'] = psnr_val
    except:
        metrics['PSNR'] = np.nan
    window_sizes = [3, 7, 11, 15]
    for win_size in window_sizes:
        try:
            if mask is not None:
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if not np.any(rows) or not np.any(cols):
                    metrics[f'SSIM_win{win_size}'] = np.nan
                    continue
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                img1_crop = img1_2d[rmin:rmax + 1, cmin:cmax + 1]
                img2_crop = img2_2d[rmin:rmax + 1, cmin:cmax + 1]
                mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]
                if img1_crop.ndim == 3:
                    _, ssim_map = ssim(img1_crop, img2_crop,
                                       data_range=255,
                                       channel_axis=2,
                                       win_size=win_size,
                                       full=True)
                    ssim_map = np.mean(ssim_map, axis=2)
                else:
                    _, ssim_map = ssim(img1_crop, img2_crop,
                                       data_range=255,
                                       win_size=win_size,
                                       full=True)
                ssim_val = np.mean(ssim_map[mask_crop])
            else:
                if img1_2d.ndim == 3:
                    ssim_val = ssim(img1_2d, img2_2d,
                                    data_range=255,
                                    channel_axis=2,
                                    win_size=win_size)
                else:
                    ssim_val = ssim(img1_2d, img2_2d,
                                    data_range=255,
                                    win_size=win_size)
            metrics[f'SSIM_win{win_size}'] = ssim_val
        except:
            metrics[f'SSIM_win{win_size}'] = np.nan
    try:
        mae = np.mean(np.abs(img1_1d - img2_1d))
        metrics['MAE'] = mae
    except:
        metrics['MAE'] = np.nan
    try:
        norm1 = np.linalg.norm(img1_1d)
        norm2 = np.linalg.norm(img2_1d)
        if norm1 == 0 or norm2 == 0:
            sam_val = np.nan
        else:
            cos_angle = np.dot(img1_1d, img2_1d) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            sam_radians = np.arccos(cos_angle)
            sam_val = np.degrees(sam_radians)
        metrics['SAM'] = sam_val
    except:
        metrics['SAM'] = np.nan
    try:
        norm1 = np.linalg.norm(img1_1d)
        norm2 = np.linalg.norm(img2_1d)
        if norm1 == 0 or norm2 == 0:
            cos_sim = np.nan
        else:
            cos_sim = np.dot(img1_1d, img2_1d) / (norm1 * norm2)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
        metrics['Cosine_Similarity'] = cos_sim
    except:
        metrics['Cosine_Similarity'] = np.nan
    return metrics
def extract_transparency_boundary(file_list, tile_width, tile_height,
                                  min_x_idx, min_y_idx, max_x_idx, max_y_idx,
                                  stride_x_px=None, stride_y_px=None):
    if stride_x_px is None:
        stride_x_px = tile_width
    if stride_y_px is None:
        stride_y_px = tile_height
    stride_x_px = int(stride_x_px)
    stride_y_px = int(stride_y_px)
    n_tiles_x = (max_x_idx - min_x_idx + 1)
    n_tiles_y = (max_y_idx - min_y_idx + 1)
    canvas_width = (n_tiles_x - 1) * stride_x_px + tile_width
    canvas_height = (n_tiles_y - 1) * stride_y_px + tile_height
    content_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
    for item in tqdm(file_list, desc="Extracting Mask from Original Alpha"):
        img = tifffile.imread(item['path'])
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            is_opaque = alpha > 0
        elif img.ndim == 3 and img.shape[0] == 4 and img.shape[1] > 4 and img.shape[2] > 4:
            alpha = img[3, :, :]
            is_opaque = alpha > 0
        else:
            if img.ndim == 2:
                is_opaque = np.ones(img.shape, dtype=bool)
            elif img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[-1]:
                is_opaque = np.ones((img.shape[1], img.shape[2]), dtype=bool)
            else:
                is_opaque = np.ones((img.shape[0], img.shape[1]), dtype=bool)
        pixel_x = (item['x_idx'] - min_x_idx) * stride_x_px
        pixel_y = (item['y_idx'] - min_y_idx) * stride_y_px
        img_h, img_w = is_opaque.shape
        y1 = min(pixel_y + img_h, canvas_height)
        x1 = min(pixel_x + img_w, canvas_width)
        content_mask[pixel_y:y1, pixel_x:x1] |= is_opaque[:(y1 - pixel_y), :(x1 - pixel_x)]
    trans_mask = ~content_mask
    boundary = binary_dilation(content_mask) & (~content_mask)
    return trans_mask, boundary
def save_colormap_png(image, output_path, colormap='viridis', dpi=300, transparent_mask=None):
    img_normalized = image.astype(np.float32) / 255.0
    cmap = cm.get_cmap(colormap)
    img_colored = cmap(img_normalized)
    img_rgba = (img_colored * 255).astype(np.uint8)
    if transparent_mask is not None:
        if transparent_mask.shape != img_rgba.shape[:2]:
            transparent_mask = transparent_mask[:img_rgba.shape[0], :img_rgba.shape[1]]
        img_rgba[transparent_mask, 3] = 0
    plt.imsave(output_path, img_rgba, dpi=dpi)
    print(f"Saved colormap PNG with transparency: {output_path}")
def save_gray_tiff_with_alpha(gray_u8, trans_mask, output_path, compression='zlib'):
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)
    H, W = gray_u8.shape
    if trans_mask is None:
        alpha = np.full((H, W), 255, dtype=np.uint8)
    else:
        if trans_mask.shape != (H, W):
            trans_mask = trans_mask[:H, :W]
        alpha = np.where(trans_mask, 0, 255).astype(np.uint8)
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    tifffile.imwrite(
        output_path,
        rgba,
        photometric='rgb',
        extrasamples=['unassalpha'],
        compression=compression
    )
def save_error_map(gt, pred, output_path, error_type='absolute', vmin=-1, vmax=1):
    gt_float = gt.astype(np.float32) / 255.0
    pred_float = pred.astype(np.float32) / 255.0
    if error_type == 'absolute':
        error = np.abs(pred_float - gt_float)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'white_to_red', ['white', 'darkred']
        )
        norm = mcolors.Normalize(vmin=0, vmax=1)
        title = 'Absolute Error Map'
    elif error_type == 'relative':
        error = np.abs(pred_float - gt_float)
        epsilon = 1e-8
        error = error / (gt_float + epsilon)
        error = np.clip(error, 0, 1)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'white_to_red', ['white', 'darkred']
        )
        norm = mcolors.Normalize(vmin=0, vmax=1)
        title = 'Relative Error Map (Absolute)'
    elif error_type == 'signed':
        error = pred_float - gt_float
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'blue_white_red', ['darkblue', 'white', 'darkred']
        )
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        title = 'Signed Error Map'
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    im = ax.imshow(error, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if error_type == 'signed':
        cbar.set_label('Error (Pred - GT)', rotation=270, labelpad=20)
    elif error_type == 'relative':
        cbar.set_label('Relative Error', rotation=270, labelpad=20)
    else:
        cbar.set_label('Absolute Error', rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error map: {output_path}")
def extract_edge_black_mask(img, threshold=5, dilate_iter=1):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    black = img <= threshold
    seed = np.zeros_like(black, dtype=bool)
    seed[0, :] = black[0, :]
    seed[-1, :] = black[-1, :]
    seed[:, 0] = black[:, 0]
    seed[:, -1] = black[:, -1]
    edge_mask = seed.copy()
    while True:
        grown = binary_dilation(edge_mask) & black
        if np.array_equal(grown, edge_mask):
            break
        edge_mask = grown
    for _ in range(dilate_iter):
        edge_mask = binary_dilation(edge_mask)
    return edge_mask
def main_predict_and_stitch(GRID_POSTPROCESS="none", stride_px=24,
                            MODEL_WEIGHTS=MODEL_WEIGHTS,
                            TARGET_ID_PREFIX="V11T17-102_D1_RNA_block1_corrected"):
    CSV_PATH = INPUT_TIFF_PATH.replace('HE', 'HE/file_mapping_with_grid.csv')
    MODEL_WEIGHTS = MODEL_WEIGHTS
    OUTPUT_DIR = INPUT_TIFF_PATH.replace('HE', 'HE/super-resolution preview')
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "individual_predictions")
    TARGET_ID_PREFIX = TARGET_ID_PREFIX
    KEEP_INDIVIDUAL_FILES = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Reading CSV: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    except:
        df = pd.read_csv(CSV_PATH, encoding='gbk')
    df_target = df[df['ID'].astype(str).str.startswith(TARGET_ID_PREFIX)].copy()
    print(f"Found {len(df_target)} samples matching prefix '{TARGET_ID_PREFIX}'")
    if len(df_target) == 0:
        print("No samples found. Exiting.")
        return
    print("\nInitializing Model...")
    model = CascadeInpaintingNet().to(device)
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS}")
        return
    print(f"Loading weights from {MODEL_WEIGHTS}...")
    try:
        ckpt = torch.load(MODEL_WEIGHTS, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
    prediction_files = []
    sampling_files = []
    original_files = []
    csv_dir = os.path.dirname(CSV_PATH)
    print("\n=== Starting Inference ===")
    print("Image processing: OriginalSize -> Model -> OriginalSize")
    with torch.no_grad():
        for idx, row in tqdm(df_target.iterrows(), total=len(df_target), desc="Predicting"):
            id_str = str(row['ID'])
            x_idx, y_idx = parse_coords_from_id(id_str)
            if x_idx is None:
                continue
            p_input = row['sampling']
            p_he = row['HE_Heatmap']
            p_original = row['original']
            if not os.path.isabs(str(p_input)):
                p_input = os.path.join(csv_dir, "sampling", os.path.basename(str(p_input)))
                p_he = os.path.join(csv_dir, "HE_Heatmap", os.path.basename(str(p_he)))
                p_original = os.path.join(csv_dir, "original", os.path.basename(str(p_original)))
            inp_tensor = load_tiff_image(p_input, force_gray=True)
            he_tensor = load_tiff_image(p_he, force_gray=False)
            if inp_tensor is None or he_tensor is None:
                if inp_tensor is None:
                    inp_tensor = load_tiff_image(
                        os.path.join(csv_dir, os.path.basename(str(row['sampling']))),
                        force_gray=True
                    )
                if he_tensor is None:
                    he_tensor = load_tiff_image(
                        os.path.join(csv_dir, os.path.basename(str(row['HE_Heatmap']))),
                        force_gray=False
                    )
                if inp_tensor is None or he_tensor is None:
                    print(f"Skipping {id_str}: Failed to load images")
                    continue
            try:
                grid_x = ast.literal_eval(row['grid_x'])
                grid_y = ast.literal_eval(row['grid_y'])
            except Exception as e:
                print(f"Skipping {id_str}: Failed to parse grid lines - {e}")
                continue
            H, W = inp_tensor.shape[1], inp_tensor.shape[2]
            boxes = compute_boxes_manual(H, W, grid_y, grid_x)
            inp_batch = inp_tensor.unsqueeze(0).to(device)
            he_batch = he_tensor.unsqueeze(0).to(device)
            boxes_batch = [boxes.to(device)]
            final_pred, _ = model(inp_batch, he_batch, [grid_y], [grid_x], boxes_batch)
            pred_img = final_pred.squeeze().detach().cpu().numpy()
            pred_img = np.clip(pred_img, 0.0, 1.0)
            pred_img = np.clip(pred_img * 255.0, 0, 255).astype(np.uint8)
            pred_img = apply_grid_postprocess(pred_img, grid_y, grid_x, method=GRID_POSTPROCESS)
            pred_filename = f"{TARGET_ID_PREFIX.strip('_')}_predicted_x{x_idx:04d}_y{y_idx:04d}.tiff"
            pred_filepath = os.path.join(PREDICTIONS_DIR, pred_filename)
            tifffile.imwrite(pred_filepath, pred_img, compression='zlib')
            prediction_files.append({
                'path': pred_filepath,
                'x_idx': x_idx,
                'y_idx': y_idx,
                'height': pred_img.shape[0],
                'width': pred_img.shape[1]
            })
            sampling_files.append({'path': p_input, 'x_idx': x_idx, 'y_idx': y_idx})
            if not os.path.exists(p_original):
                p_original = os.path.join(csv_dir, os.path.basename(str(row['original'])))
            if os.path.exists(p_original):
                original_files.append({'path': p_original, 'x_idx': x_idx, 'y_idx': y_idx})
    print(f"\n{len(prediction_files)} predictions completed.")
    if not prediction_files:
        print("No valid predictions generated. Exiting.")
        return
    tile_height = prediction_files[0]['height']
    tile_width = prediction_files[0]['width']
    min_x_idx = min(p['x_idx'] for p in prediction_files)
    max_x_idx = max(p['x_idx'] for p in prediction_files)
    min_y_idx = min(p['y_idx'] for p in prediction_files)
    max_y_idx = max(p['y_idx'] for p in prediction_files)
    print(f"\nTile size: {tile_width} x {tile_height}")
    print(f"Index range: X[{min_x_idx}, {max_x_idx}], Y[{min_y_idx}, {max_y_idx}]")
    print("\n=== Stitching Predicted Images ===")
    canvas_predicted = stitch_images(
        prediction_files, tile_width, tile_height,
        min_x_idx, min_y_idx, max_x_idx, max_y_idx,
        stride_x_px=stride_px, stride_y_px=stride_px
    )
    print("\n=== Stitching Sampling Images ===")
    canvas_sampling = stitch_images(
        sampling_files, tile_width, tile_height,
        min_x_idx, min_y_idx, max_x_idx, max_y_idx,
        stride_x_px=stride_px, stride_y_px=stride_px
    )
    if original_files:
        print("\n=== Stitching Original Images ===")
        canvas_original = stitch_images(
            original_files, tile_width, tile_height,
            min_x_idx, min_y_idx, max_x_idx, max_y_idx,
            stride_x_px=stride_px, stride_y_px=stride_px
        )
    else:
        canvas_original = None
        print("\n=== No Original Images Found ===")
    print("\n=== Extracting Transparency Boundary from Original (GT) ===")
    if len(original_files) == 0:
        raise RuntimeError("Original files are required because all masks must come from Original alpha.")
    trans_mask, boundary = extract_transparency_boundary(
        original_files, tile_width, tile_height,
        min_x_idx, min_y_idx, max_x_idx, max_y_idx,
        stride_x_px=stride_px, stride_y_px=stride_px
    )
    edge_black_mask = extract_edge_black_mask(
        canvas_original,
        threshold=5,
        dilate_iter=1
    )
    trans_mask = trans_mask | edge_black_mask
    canvas_predicted[trans_mask] = 0
    canvas_sampling[trans_mask] = 0
    if canvas_original is not None:
        canvas_original[trans_mask] = 0
    print("\n=== Saving Color-mapped PNGs with Transparency ===")
    predicted_png_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Predicted.png")
    save_colormap_png(canvas_predicted, predicted_png_path, colormap='rainbow', dpi=300, transparent_mask=trans_mask)
    input_png_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Input.png")
    save_colormap_png(canvas_sampling, input_png_path, colormap='rainbow', dpi=300, transparent_mask=trans_mask)
    if canvas_original is not None:
        original_png_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Original.png")
        save_colormap_png(canvas_original, original_png_path, colormap='rainbow', dpi=300, transparent_mask=trans_mask)
    predicted_path_tiff = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Predicted.tiff")
    save_gray_tiff_with_alpha(canvas_predicted, trans_mask, predicted_path_tiff, compression='zlib')
    input_path_tiff = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Input.tiff")
    save_gray_tiff_with_alpha(canvas_sampling, trans_mask, input_path_tiff, compression='zlib')
    if canvas_original is not None:
        original_path_tiff = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Original.tiff")
        save_gray_tiff_with_alpha(canvas_original, trans_mask, original_path_tiff, compression='zlib')
    if canvas_original is not None:
        print("\n=== Generating Error Maps ===")
        abs_error_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Error_Absolute.png")
        save_error_map(canvas_original, canvas_predicted, abs_error_path,
                       error_type='absolute', vmin=0, vmax=1)
        rel_error_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Error_Relative.png")
        save_error_map(canvas_original, canvas_predicted, rel_error_path,
                       error_type='relative', vmin=0, vmax=1)
        signed_error_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_Error_Signed.png")
        save_error_map(canvas_original, canvas_predicted, signed_error_path,
                       error_type='signed', vmin=-1, vmax=1)
    print("\n=== Calculating Metrics ===")
    valid_mask = ~trans_mask
    if canvas_original is not None:
        print("\n1. Original vs Predicted:")
        metrics_orig_pred = calculate_metrics(canvas_original, canvas_predicted, valid_mask)
        for metric, value in metrics_orig_pred.items():
            print(f"   {metric}: {value:.4f}")
        metrics_df = pd.DataFrame({
            'Metric': list(metrics_orig_pred.keys()),
            'Original_vs_Predicted': list(metrics_orig_pred.values())
        })
        metrics_csv_path = os.path.join(OUTPUT_DIR, f"{TARGET_ID_PREFIX.strip('_')}_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nMetrics saved to: {metrics_csv_path}")
    else:
        print("\n=== Cannot calculate metrics: No original images available ===")
    if not KEEP_INDIVIDUAL_FILES:
        print("\n=== Cleaning up individual prediction files ===")
        for p in prediction_files:
            try:
                os.remove(p['path'])
            except Exception as e:
                print(f"Failed to remove {p['path']}: {e}")
    print("\n=== Processing Complete ===")
    print(f"Output directory: {OUTPUT_DIR}")
main_predict_and_stitch(GRID_POSTPROCESS="none", TARGET_ID_PREFIX='', MODEL_WEIGHTS=MODEL_WEIGHTS)
