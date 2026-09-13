import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import lpips
from ntn.dct_util import dct_2d, idct_2d, low_pass, high_pass
from ntn.ddim_sampler import DDIM_Sampler
from adversarial_guidance import AdversarialGuidance
import os  # [NEW] 用于创建目录
import torchvision.utils as vutils  # [NEW] 用于保存图像

# 修复功能依赖检查
try:
    from scipy.spatial import KDTree
    import cv2

    SCIPY_CV2_AVAILABLE = True
    HAS_XIMGPROC = hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter')
except ImportError:
    SCIPY_CV2_AVAILABLE = False
    HAS_XIMGPROC = False
    print("⚠ 修复功能需要 scipy, opencv-python, opencv-contrib-python。")
    print("  请安装：pip install scipy opencv-python opencv-contrib-python")


# ========================== EdgeGuidance ==========================
class EdgeGuidance:
    def __init__(self):
        self.laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)
        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)

    def extract_edges(self, image, use_sobel=False):
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
        else:
            gray = image
        if use_sobel:
            sobel_x = self.sobel_x.to(image.device)
            sobel_y = self.sobel_y.to(image.device)
            edge_x = F.conv2d(gray, sobel_x, padding=1)
            edge_y = F.conv2d(gray, sobel_y, padding=1)
            edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        else:
            laplacian = self.laplacian_kernel.to(image.device)
            edges = F.conv2d(gray, laplacian, padding=1)
            edges = torch.abs(edges)
        edges = torch.sqrt(edges + 1e-8)
        edges = edges / (edges.max() + 1e-8)
        return edges

    def get_edge_weight(self, latent, model, mask, target_size=None):
        with torch.no_grad():
            image = model.decode_first_stage(latent)
            image = torch.clamp((image + 1) / 2, 0.0, 1.0)
            edges = self.extract_edges(image, use_sobel=True)
            if mask is not None:
                if mask.shape[-2:] != edges.shape[-2:]:
                    mask_resized = F.interpolate(mask, size=edges.shape[-2:], mode='bilinear', align_corners=True)
                else:
                    mask_resized = mask
                edge_weight = edges * mask_resized
            else:
                edge_weight = edges
        edge_weight = torch.clamp(edge_weight, 0.0, 1.0)
        edge_weight = edge_weight ** 0.5
        if target_size is not None and edge_weight.shape[-2:] != target_size:
            edge_weight = F.interpolate(edge_weight, size=target_size, mode='bilinear', align_corners=True)
        return edge_weight

    def guided_fusion(self, low_freq, high_freq, edge_weight):
        if edge_weight.shape[1] != low_freq.shape[1]:
            edge_weight = edge_weight.repeat(1, low_freq.shape[1], 1, 1)
        merged = low_freq * (1 - edge_weight) + high_freq * edge_weight
        return merged


# ========================== EmbeddingSpaceOptimizer ==========================
class EmbeddingSpaceOptimizer:
    def __init__(self, clip_model, device, steps=3, lr=0.003,
                 color_weight=0.1, adaptive_color=True,
                 sensitivity=3.0, boost_factor=2.0, low_loss_threshold=0.2,
                 local_band_width=3):
        self.clip_model = clip_model
        self.device = device
        self.steps = steps
        self.lr = lr
        self.base_color_weight = color_weight
        self.adaptive_color = adaptive_color
        self.sensitivity = sensitivity
        self.boost_factor = boost_factor
        self.low_loss_threshold = low_loss_threshold
        self.local_band_width = local_band_width
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.original_image_cache = None

    def preprocess_image_for_clip(self, image):
        image_normalized = (image + 1) / 2
        image_normalized = torch.clamp(image_normalized, 0, 1)
        image_resized = F.interpolate(image_normalized, size=(224, 224), mode='bilinear', align_corners=False)
        image_norm = (image_resized - self.mean) / self.std
        return image_norm

    def _get_local_known_mask(self, mask, band_width=3):
        kernel_size = 2 * band_width + 1
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=band_width)
        band = dilated - mask
        return band

    def _compute_color_statistics_local(self, image, mask, band_width):
        band = self._get_local_known_mask(mask, band_width)
        band = band.expand(-1, 3, -1, -1)
        numerator = (image * band).sum(dim=[2, 3], keepdim=True)
        denominator = band.sum(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        mean_band = numerator / denominator
        diff_sq = (image - mean_band) ** 2
        numerator_std = (diff_sq * band).sum(dim=[2, 3], keepdim=True)
        std_band = torch.sqrt(numerator_std / denominator + 1e-8)
        return mean_band, std_band

    def _prepare_original_stats(self, model, original_latent, mask, band_width):
        if self.original_image_cache is not None:
            return self.original_image_cache
        with torch.no_grad():
            original_image = model.decode_first_stage(original_latent)
            original_image = torch.clamp((original_image + 1) / 2, 0, 1)
            orig_mean, orig_std = self._compute_color_statistics_local(original_image, mask, band_width)
            self.original_image_cache = (orig_mean, orig_std)
        return self.original_image_cache

    def optimize_latent(self, latent, text_embedding, model,
                        mask=None, original_latent=None,
                        original_image=None, color_weight=None,
                        adaptive_color=None, sensitivity=None,
                        boost_factor=None, low_loss_threshold=None,
                        local_band_width=None):
        use_adaptive = adaptive_color if adaptive_color is not None else self.adaptive_color
        cw = color_weight if color_weight is not None else self.base_color_weight
        sens = sensitivity if sensitivity is not None else self.sensitivity
        boost = boost_factor if boost_factor is not None else self.boost_factor
        thresh = low_loss_threshold if low_loss_threshold is not None else self.low_loss_threshold
        band_w = local_band_width if local_band_width is not None else self.local_band_width
        mask_bin = (mask > 0.5).float() if mask is not None else None
        orig_mean, orig_std = None, None
        if original_latent is not None and mask_bin is not None:
            orig_mean, orig_std = self._prepare_original_stats(model, original_latent, mask_bin, band_w)
        elif original_image is not None and mask_bin is not None:
            with torch.no_grad():
                original_image_01 = (original_image + 1) / 2 if original_image.min() < 0 else original_image
                orig_mean, orig_std = self._compute_color_statistics_local(
                    torch.clamp(original_image_01, 0, 1), mask_bin, band_w)
        latent_opt = latent.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([latent_opt], lr=self.lr, momentum=0.9)
        text_norm = F.normalize(text_embedding, dim=-1)
        for step in range(self.steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                image = model.decode_first_stage(latent_opt)
                if not image.requires_grad:
                    image = image.clone().detach().requires_grad_(True)
                image_norm = self.preprocess_image_for_clip(image)
                image_features = self.clip_model.encode_image(image_norm)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features * text_norm).sum(dim=-1)
                clip_loss = (1 - similarity).mean()
                total_loss = clip_loss
                if orig_mean is not None and mask_bin is not None:
                    if use_adaptive:
                        if clip_loss.detach() < thresh:
                            effective_cw = cw * boost
                            total_loss = 0.01 * clip_loss
                        else:
                            loss_scale = torch.exp(-clip_loss.detach() * sens)
                            effective_cw = cw * loss_scale
                            total_loss = clip_loss
                    else:
                        effective_cw = cw
                        total_loss = clip_loss
                    if effective_cw > 1e-6:
                        cur_image = torch.clamp((image + 1) / 2, 0, 1)
                        cur_mean, cur_std = self._compute_color_statistics_local(cur_image, mask_bin, band_w)
                        color_loss = F.l1_loss(cur_mean, orig_mean) + F.l1_loss(cur_std, orig_std)
                        total_loss = total_loss + effective_cw * color_loss
                if mask is not None and original_latent is not None:
                    mask_resized = F.interpolate(mask, size=latent_opt.shape[-2:], mode='nearest')
                    consistency_loss = F.mse_loss(
                        latent_opt * (1 - mask_resized),
                        original_latent * (1 - mask_resized)
                    )
                    total_loss = total_loss + 0.1 * consistency_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([latent_opt], max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                if mask is not None and original_latent is not None:
                    mask_resized = F.interpolate(mask, size=latent_opt.shape[-2:], mode='nearest')
                    latent_opt = latent_opt * mask_resized + original_latent * (1 - mask_resized)
        return latent_opt.detach()


# ========================== Ntn_Sampler ==========================
class Ntn_Sampler(DDIM_Sampler):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.adversarial_guidance = None
        self.edge_guider = None
        self.embedding_optimizer = None
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.model.device)
        print("✓ LPIPS 感知损失已初始化 (AlexNet)")

    def init_adversarial_guidance(self, discriminator, guidance_weight=0.3):
        self.adversarial_guidance = AdversarialGuidance(discriminator, self.model, guidance_weight)
        print(f"✓ 对抗性引导已初始化，权重: {guidance_weight}")

    def init_edge_guidance(self):
        self.edge_guider = EdgeGuidance()
        print("✓ 边缘引导已初始化")

    def init_embedding_optimizer(self, clip_model, device, steps=3, lr=0.003,
                                 color_weight=0.1, adaptive_color=True, sensitivity=3.0):
        self.embedding_optimizer = EmbeddingSpaceOptimizer(
            clip_model, device, steps, lr, color_weight, adaptive_color, sensitivity
        )
        print(f"✓ 嵌入空间优化器已初始化 (steps={steps}, lr={lr})")

    # ---------- 辅助函数 ----------
    def smooth_mask(self, mask, sigma=0.5):
        if sigma <= 0:
            return mask
        device = mask.device
        dtype = mask.dtype
        kernel_size = int(2 * sigma * 3 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
        kernel = gauss_2d.view(1, 1, kernel_size, kernel_size)
        if mask.shape[1] != 1:
            mask = mask[:, 0:1, :, :]
        smoothed = F.conv2d(mask, kernel, padding=kernel_size // 2)
        return smoothed

    def spatial_blend(self, latent1, latent2, mask, feather_radius=5):
        sigma = feather_radius / 2.0
        alpha = self.smooth_mask(mask, sigma=sigma)
        if alpha.shape[1] != latent1.shape[1]:
            alpha = alpha.repeat(1, latent1.shape[1], 1, 1)
        blended = latent1 * (1 - alpha) + latent2 * alpha
        return blended

    # ---------- 修复函数 ----------
    def remove_small_blobs(self, image_tensor, brightness_thresh=0.85,
                           max_area=80, blur_radius=1.5, strength=0.7):
        if not SCIPY_CV2_AVAILABLE:
            return image_tensor
        B, C, H, W = image_tensor.shape
        result = image_tensor.clone()
        for b in range(B):
            img_np = (result[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            bright_mask = (gray > brightness_thresh).astype(np.uint8) * 255
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
            blob_mask = np.zeros_like(bright_mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area <= max_area:
                    blob_mask[labels == i] = 255
            if np.sum(blob_mask) == 0:
                continue
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(blob_mask, kernel, iterations=1)
            blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=blur_radius)
            alpha = strength * (dilated / 255.0).astype(np.float32)
            alpha = np.expand_dims(alpha, axis=2)
            blended = (img_np.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha).astype(np.uint8)
            if HAS_XIMGPROC:
                guide = blended.astype(np.float32) / 255.0
                filtered = cv2.ximgproc.guidedFilter(guide, guide, radius=2, eps=1e-4)
                blended = (filtered * 255).astype(np.uint8)
            result[b] = torch.from_numpy(blended.transpose(2, 0, 1)).to(image_tensor.device) / 255.0
        return result

    def fix_overflow_pixels(self, image_tensor, mask=None,
                            low_thresh=0.03, high_thresh=0.97,
                            min_area=5, max_area=500,
                            inpaint_radius=5):
        if not SCIPY_CV2_AVAILABLE:
            return image_tensor
        B, C, H, W = image_tensor.shape
        repaired = image_tensor.clone()
        if mask is not None:
            if mask.shape[-2:] != (H, W):
                mask_resized = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=True)
            else:
                mask_resized = mask
        else:
            mask_resized = None
        for b in range(B):
            img_np = (repaired[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_float = img_np.astype(np.float32) / 255.0
            overflow_mask = (img_float < low_thresh) | (img_float > high_thresh)
            overflow_mask = np.any(overflow_mask, axis=2).astype(np.uint8) * 255
            if mask_resized is not None:
                mask_np = (mask_resized[b, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                if mask_np.shape != overflow_mask.shape:
                    mask_np = cv2.resize(mask_np, (overflow_mask.shape[1], overflow_mask.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                overflow_mask = cv2.bitwise_and(overflow_mask, mask_np)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(overflow_mask, connectivity=8)
            final_overflow = np.zeros_like(overflow_mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    final_overflow[labels == i] = 255
            if np.sum(final_overflow) == 0:
                continue
            kernel = np.ones((3, 3), np.uint8)
            overflow_dilated = cv2.dilate(final_overflow, kernel, iterations=1)
            inpainted = cv2.inpaint(img_np, overflow_dilated, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
            repaired_np = inpainted.astype(np.float32) / 255.0
            repaired[b] = torch.from_numpy(repaired_np.transpose(2, 0, 1)).to(image_tensor.device)
        return repaired

    def repair_artifacts(self, image_tensor, original_mask,
                         black_thresh=20, color_diff_thresh=25,
                         sat_thresh=0.2, max_artifact_area=300,
                         iterations=5, use_inpaint=True):
        if not SCIPY_CV2_AVAILABLE:
            return image_tensor
        B, C, H, W = image_tensor.shape
        repaired = image_tensor.clone()
        if original_mask.shape[-2:] != (H, W):
            original_mask_resized = F.interpolate(original_mask, size=(H, W), mode='nearest')
        else:
            original_mask_resized = original_mask
        for b in range(B):
            img_np = (repaired[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            known_np = (original_mask_resized[b, 0].cpu().numpy() < 0.5).astype(np.uint8)
            for _ in range(iterations):
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                r, g, b_chan = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
                black_candidate = (gray < black_thresh) & (r < black_thresh + 20) & (g < black_thresh + 20) & (
                            b_chan < black_thresh + 20)
                median = cv2.medianBlur(img_np, 5)
                diff = np.abs(img_np.astype(np.int16) - median.astype(np.int16)).max(axis=2)
                max_rgb = np.max(img_np, axis=2)
                min_rgb = np.min(img_np, axis=2)
                saturation = (max_rgb - min_rgb) / (max_rgb + 1e-5)
                color_noise = (diff > color_diff_thresh) & (saturation > sat_thresh)
                artifact_mask = (black_candidate | color_noise).astype(np.uint8) * 255
                kernel = np.ones((7, 7), np.uint8)
                artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(artifact_mask, connectivity=8)
                final_mask = np.zeros_like(artifact_mask)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] <= max_artifact_area:
                        final_mask[labels == i] = 255
                artifact_mask = final_mask
                if np.sum(artifact_mask) == 0:
                    break
                artifact_mask_dilated = cv2.dilate(artifact_mask, kernel, iterations=2)
                if use_inpaint and cv2.__version__.startswith('4'):
                    inpainted = cv2.inpaint(img_np, artifact_mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                    img_np = inpainted
                else:
                    normal_generated = (artifact_mask == 0) & (original_mask_resized[b, 0].cpu().numpy() >= 0.5)
                    valid_mask = (known_np == 1) | normal_generated
                    if np.sum(valid_mask) == 0:
                        valid_mask = known_np
                        if np.sum(valid_mask) == 0:
                            break
                    valid_mask_dilated = cv2.dilate(valid_mask.astype(np.uint8), np.ones((9, 9), np.uint8),
                                                    iterations=2).astype(bool)
                    valid_coords = np.argwhere(valid_mask_dilated)
                    valid_colors = img_np[valid_coords[:, 0], valid_coords[:, 1], :]
                    defect_coords = np.argwhere(artifact_mask_dilated)
                    if len(defect_coords) > 0 and len(valid_coords) > 0:
                        tree = KDTree(valid_coords)
                        _, indices = tree.query(defect_coords, k=1)
                        nearest_colors = valid_colors[indices]
                        for (y, x), color in zip(defect_coords, nearest_colors):
                            img_np[y, x, :] = color
                if np.any(artifact_mask) and HAS_XIMGPROC:
                    guide = img_np.astype(np.float32) / 255.0
                    filtered = cv2.ximgproc.guidedFilter(guide, guide, radius=2, eps=1e-4)
                    img_np = (filtered * 255).astype(np.uint8)
            repaired_np = img_np.astype(np.float32) / 255.0
            repaired[b] = torch.from_numpy(repaired_np.transpose(2, 0, 1)).to(image_tensor.device)
        return repaired

    def suppress_exploding_pixels(self, image_tensor, mask=None,
                                  brightness_thresh=0.92,
                                  color_imbalance=0.6,
                                  local_window=5,
                                  local_ratio=0.35,
                                  min_area=2, max_area=200,
                                  use_inpaint=True, inpaint_radius=3):
        if not SCIPY_CV2_AVAILABLE:
            return image_tensor
        B, C, H, W = image_tensor.shape
        repaired = image_tensor.clone()
        mask_resized = None
        if mask is not None:
            if mask.shape[-2:] != (H, W):
                mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
            else:
                mask_resized = mask
        for b in range(B):
            img_np = (repaired[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_float = img_np.astype(np.float32) / 255.0
            max_ch = np.max(img_float, axis=2)
            min_ch = np.min(img_float, axis=2)
            too_bright = max_ch > brightness_thresh
            color_imbalanced = (max_ch - min_ch) > color_imbalance
            median = cv2.medianBlur(img_np, local_window) / 255.0
            local_diff = np.abs(img_float - median).max(axis=2)
            isolated = local_diff > local_ratio
            anomaly = (too_bright & color_imbalanced) & isolated
            if mask_resized is not None:
                mask_np = (mask_resized[b, 0].cpu().numpy() > 0.5)
                anomaly = anomaly & mask_np
            anomaly_uint8 = anomaly.astype(np.uint8) * 255
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(anomaly_uint8, connectivity=8)
            final_mask = np.zeros_like(anomaly_uint8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    final_mask[labels == i] = 255
            if np.sum(final_mask) == 0:
                continue
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)
            if use_inpaint and cv2.__version__.startswith('4'):
                repaired_np = cv2.inpaint(img_np, final_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
            else:
                median_filtered = cv2.medianBlur(img_np, 5)
                repaired_np = img_np.copy()
                repaired_np[final_mask > 0] = median_filtered[final_mask > 0]
            repaired[b] = torch.from_numpy(repaired_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(
                image_tensor.device)
        return repaired

    def repair_white_regions(self, image_tensor, mask=None,
                             brightness_thresh=0.90,
                             min_area=200, max_area=20000,
                             inpaint_radius=5):
        if not SCIPY_CV2_AVAILABLE:
            return image_tensor
        B, C, H, W = image_tensor.shape
        repaired = image_tensor.clone()
        mask_resized = None
        if mask is not None:
            if mask.shape[-2:] != (H, W):
                mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
            else:
                mask_resized = mask
        for b in range(B):
            img_np = (repaired[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            white_mask = (gray > brightness_thresh).astype(np.uint8) * 255
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
            region_mask = np.zeros_like(white_mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    region_mask[labels == i] = 255
            if mask_resized is not None:
                mask_np = (mask_resized[b, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                region_mask = cv2.bitwise_and(region_mask, mask_np)
            if np.sum(region_mask) == 0:
                continue
            kernel = np.ones((5, 5), np.uint8)
            region_mask = cv2.dilate(region_mask, kernel, iterations=1)
            repaired_np = cv2.inpaint(img_np, region_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
            repaired[b] = torch.from_numpy(repaired_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(
                image_tensor.device)
        return repaired

    def repair_white_patches(self, image_tensor, mask=None,
                             brightness_thresh=0.92,
                             saturation_thresh=40,
                             min_area=50, max_area=2000,
                             inpaint_radius=5):
        if not SCIPY_CV2_AVAILABLE:
            return image_tensor
        B, C, H, W = image_tensor.shape
        repaired = image_tensor.clone()
        mask_resized = None
        if mask is not None:
            if mask.shape[-2:] != (H, W):
                mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
            else:
                mask_resized = mask
        for b in range(B):
            img_np = (repaired[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            v = hsv[:, :, 2]
            s = hsv[:, :, 1]
            white_mask = (v > int(brightness_thresh * 255)) & (s < saturation_thresh)
            white_mask = white_mask.astype(np.uint8) * 255
            if mask_resized is not None:
                m_np = (mask_resized[b, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                white_mask = cv2.bitwise_and(white_mask, m_np)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
            final_mask = np.zeros_like(white_mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    final_mask[labels == i] = 255
            if np.sum(final_mask) == 0:
                continue
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)
            repaired_np = cv2.inpaint(img_np, final_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
            repaired[b] = torch.from_numpy(repaired_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(
                image_tensor.device)
        return repaired

    def remove_white_blobs(self, image_tensor, mask_tensor,
                           brightness_thresh=0.92,
                           saturation_thresh=50,
                           min_area=20,
                           max_area=2000,
                           valid_include_known=True,
                           debug_save_path=None):
        if not SCIPY_CV2_AVAILABLE:
            print("⚠ 需要 scipy 和 opencv-python，跳过白片移除")
            return image_tensor

        B, C, H, W = image_tensor.shape
        repaired = image_tensor.clone()

        if mask_tensor.shape[-2:] != (H, W):
            mask_resized = F.interpolate(mask_tensor, size=(H, W), mode='nearest')
        else:
            mask_resized = mask_tensor

        for b in range(B):
            img_np = (repaired[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            mask_np = (mask_resized[b, 0].cpu().numpy() > 0.5).astype(np.uint8)  # 已知=1,修复=0

            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            v = hsv[:, :, 2] / 255.0
            s = hsv[:, :, 1]
            white_candidate = (v > brightness_thresh) & (s < saturation_thresh)

            r = img_np[:, :, 0] / 255.0
            g = img_np[:, :, 1] / 255.0
            b_ch = img_np[:, :, 2] / 255.0
            max_rgb = np.maximum(np.maximum(r, g), b_ch)
            min_rgb = np.minimum(np.minimum(r, g), b_ch)
            color_range = max_rgb - min_rgb
            white_candidate = white_candidate & (color_range < 0.15)

            white_mask = (white_candidate & (mask_np == 0)).astype(np.uint8)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
            final_mask = np.zeros_like(white_mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    final_mask[labels == i] = 255

            if np.sum(final_mask) == 0:
                print(f"  白片检测: 未发现符合条件的白片 (面积 {min_area}~{max_area})")
                continue
            else:
                print(f"  白片检测: 发现 {num_labels - 1} 个连通域，保留 {np.sum(final_mask > 0)} 像素")

            if valid_include_known:
                effective_mask = (mask_np == 1) | ((mask_np == 0) & (final_mask == 0))
            else:
                effective_mask = ((mask_np == 0) & (final_mask == 0))

            y_eff, x_eff = np.where(effective_mask)
            if len(y_eff) == 0:
                print(f"  警告: 没有有效像素可用，跳过替换")
                continue

            eff_coords = np.stack([x_eff, y_eff], axis=1)
            eff_colors = img_np[y_eff, x_eff, :]

            white_y, white_x = np.where(final_mask)
            white_coords = np.stack([white_x, white_y], axis=1)
            tree = KDTree(eff_coords)
            distances, indices = tree.query(white_coords, k=1)

            new_img = img_np.copy()
            for (wy, wx), idx in zip(zip(white_y, white_x), indices):
                new_img[wy, wx] = eff_colors[idx]

            if debug_save_path is not None:
                debug_mask = final_mask
                cv2.imwrite(f"{debug_save_path}_b{b}.png", debug_mask)

            repaired[b] = torch.from_numpy(new_img.astype(np.float32) / 255.0).permute(2, 0, 1).to(image_tensor.device)

        return repaired

    # ---------- 约束方法 ----------
    def apply_embedding_optimization(self, latent, text_embedding, mask, original_latent, progress):
        if self.embedding_optimizer is None:
            return latent
        if progress > 0.3:
            return self.embedding_optimizer.optimize_latent(latent, text_embedding, self.model, mask, original_latent)
        return latent

    def _get_intermediate_step(self, intermediates, intermediate_steps, target_step, mask, latent):
        target_step_val = target_step.item() if torch.is_tensor(target_step) else target_step
        if target_step_val in intermediate_steps:
            step_idx = intermediate_steps.index(target_step_val)
        else:
            closest_step = min(intermediate_steps, key=lambda x: abs(x - target_step_val))
            step_idx = intermediate_steps.index(closest_step)
        return intermediates[step_idx] * (1 - mask) + latent * mask

    def compute_ssim_loss(self, img1, img2, mask=None):
        try:
            if mask is not None:
                if mask.shape[-2:] != img1.shape[-2:]:
                    mask = F.interpolate(mask, size=img1.shape[-2:], mode='nearest')
                if mask.shape[1] == 1 and img1.shape[1] == 3:
                    mask = mask.repeat(1, 3, 1, 1)
                img1 = img1 * mask
                img2 = img2 * mask
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            mu1 = F.avg_pool2d(img1, 3, 1, 1)
            mu2 = F.avg_pool2d(img2, 3, 1, 1)
            sigma1 = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1 ** 2
            sigma2 = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2 ** 2
            sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                    (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
            return 1 - ssim_map.mean()
        except Exception:
            return torch.tensor(0.0, device=img1.device)

    def apply_ssim_constraint(self, latent, original_latent, mask, step_weight=0.05):
        try:
            latent_with_grad = latent.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                current_image = self.model.decode_first_stage(latent_with_grad)
                current_image = torch.clamp((current_image + 1) / 2, 0.0, 1.0)
            with torch.no_grad():
                original_image = self.model.decode_first_stage(original_latent)
                original_image = torch.clamp((original_image + 1) / 2, 0.0, 1.0)
            ssim_loss = self.compute_ssim_loss(current_image, original_image, mask)
            if torch.isnan(ssim_loss):
                return latent
            gradient = torch.autograd.grad(ssim_loss, latent_with_grad, allow_unused=True, retain_graph=False)[0]
            if gradient is None:
                return latent
            grad_norm = gradient.norm()
            if grad_norm > 0:
                gradient = gradient / (grad_norm + 1e-8)
            return (latent - step_weight * gradient).detach()
        except Exception:
            return latent

    def apply_lpips_constraint(self, latent, original_latent, mask, step_weight=0.02):
        try:
            latent_grad = latent.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                current_image = self.model.decode_first_stage(latent_grad)
                current_image = torch.clamp((current_image + 1) / 2, 0.0, 1.0)
            with torch.no_grad():
                original_image = self.model.decode_first_stage(original_latent)
                original_image = torch.clamp((original_image + 1) / 2, 0.0, 1.0)
            if mask.shape[-2:] != current_image.shape[-2:]:
                mask_resized = F.interpolate(mask, size=current_image.shape[-2:], mode='nearest')
            else:
                mask_resized = mask
            known_mask = (1 - mask_resized).float()
            if known_mask.shape[1] == 1 and current_image.shape[1] == 3:
                known_mask = known_mask.repeat(1, 3, 1, 1)
            lpips_val = self.lpips_loss(current_image * known_mask, original_image * known_mask)
            loss = lpips_val.mean()
            if torch.isnan(loss):
                return latent
            gradient = torch.autograd.grad(loss, latent_grad)[0]
            if gradient is None:
                return latent
            grad_norm = gradient.norm()
            if grad_norm > 0:
                gradient = gradient / (grad_norm + 1e-8)
            return (latent - step_weight * gradient).detach()
        except Exception:
            return latent

    # ---------- 主采样函数 decode（新增保存中间结果参数） ----------
    @torch.no_grad()
    def decode(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
               unconditional_conditioning, mask, unmask, conds, conds1,
               use_original_steps=False, callback=None,
               threshold=3, end_step=500,
               use_adversarial=False,
               use_edge_guidance=False,
               use_ssim_constraint=False,
               use_lpips_constraint=False,
               original_latent=None,
               text_embedding=None,
               use_text_guidance=False,
               use_repair=False,
               use_overflow_fix=False,
               use_white_blob_removal=False,
               use_exploding_suppress=True,
               use_white_region_repair=True,
               use_white_patch_repair=False,
               use_spatial_fusion=True,
               feather_radius=9,
               # 溢出修复参数
               overflow_low_thresh=0.03,
               overflow_high_thresh=0.97,
               overflow_min_area=5,
               overflow_max_area=500,
               overflow_inpaint_radius=5,
               # 小斑点去除参数
               white_blob_brightness_thresh=0.85,
               white_blob_max_area=80,
               white_blob_blur_radius=1.5,
               white_blob_strength=0.7,
               # 爆炸点抑制参数
               explode_brightness_thresh=0.92,
               explode_color_imbalance=0.6,
               explode_local_window=5,
               explode_local_ratio=0.35,
               explode_min_area=2,
               explode_max_area=200,
               explode_inpaint_radius=3,
               # 白色片状区域修复参数（原有）
               white_region_brightness_thresh=0.90,
               white_region_min_area=200,
               white_region_max_area=20000,
               white_region_inpaint_radius=5,
               # 新增白片修复参数
               white_patch_brightness_thresh=0.92,
               white_patch_saturation_thresh=40,
               white_patch_min_area=50,
               white_patch_max_area=2000,
               white_patch_inpaint_radius=5,
               # 常规修复参数
               repair_black_thresh=20,
               repair_color_sensitivity=25,
               repair_sat_thresh=0.2,
               repair_max_area=300,
               repair_iterations=5,
               repair_use_inpaint=True,
               # 其他调优
               edge_guidance_weight=5.0,
               ssim_weight=0.10,
               lpips_weight=0.05,
               dct_aggressive=True,
               return_latent=False,
               # [NEW] 保存中间结果的参数
               save_latents_dir=None,  # 保存 latent 的文件夹路径
               save_images_dir=None,  # 保存生成图像的文件夹路径
               save_every=1):  # 每 N 步保存一次（1 表示每一步都保存）
        """
        主采样函数。保留平滑掩码和羽化半径9，将引导更新频率从5步恢复为3步。
        新增 save_latents_dir, save_images_dir, save_every 用于保存中间结果。
        """
        if use_edge_guidance and self.edge_guider is None:
            self.init_edge_guidance()
        if use_text_guidance and text_embedding is not None and self.embedding_optimizer is None:
            try:
                import clip
                print("\n[嵌入优化] 正在初始化...")
                clip_model, _ = clip.load("ViT-B/32", device=text_embedding.device)
                self.init_embedding_optimizer(clip_model, text_embedding.device, steps=4, lr=0.001, color_weight=0.05)
            except Exception as e:
                print(f"⚠ 嵌入优化器初始化失败: {e}")

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(
            f"DDIM Sampling {total_steps} steps | RegionRepair:{use_white_region_repair} PatchRepair:{use_white_patch_repair} ExplodeSuppress:{use_exploding_suppress} | "
            f"Spatial Fusion:{use_spatial_fusion} Edge:{use_edge_guidance} SSIM:{use_ssim_constraint} LPIPS:{use_lpips_constraint} | "
            f"Return latent: {return_latent}")

        if use_adversarial and self.adversarial_guidance is None:
            print("⚠ 对抗性引导未初始化")

        if use_text_guidance and text_embedding is not None:
            cond_with_text = cond.copy()
            cond_with_text['text_embedding'] = text_embedding
            conds_with_text = conds.copy()
            conds_with_text['text_embedding'] = text_embedding
        else:
            cond_with_text = cond
            conds_with_text = conds

        x_dec = torch.randn_like(ref_latent)
        ref_latent1 = torch.randn_like(ref_latent)

        total_elements = mask.numel()
        zero_ratio = (mask == 0).sum().item() / total_elements
        mask_inpainting = self.smooth_mask(mask, sigma=0.5)

        # 保留平滑掩码（sigma=1.0）
        mask_smooth = self.smooth_mask(mask, sigma=1.0)

        h, w = ref_latent.shape[-2:]
        max_freq = h + w - 2

        if not use_spatial_fusion:
            if dct_aggressive:
                threshold_val = min(30 + 5 * zero_ratio, max_freq - 1)
                threshold1 = min(max(0, 3 - 2 * zero_ratio), max_freq - 2)
                threshold2 = min(threshold1 + 2, min(80 + 10 * zero_ratio, max_freq - 1))
            else:
                threshold_val = min(50 + 10 * zero_ratio, max_freq - 1)
                threshold1 = min(max(0, 5 - 5 * zero_ratio), max_freq - 2)
                threshold2 = min(threshold1 + 2, min(150 + 20 * zero_ratio, max_freq - 1))
            threshold2 = max(threshold2, threshold1 + 2)

        intermediate_steps = unmask['intermediate_steps']
        intermediates = unmask['intermediates']
        last_optimization_step = -5

        iterator = tqdm(time_range, desc='Decoding', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)
            progress = i / total_steps

            if step >= end_step:
                ref_latent = self._get_intermediate_step(intermediates, intermediate_steps, ts + 1, mask, ref_latent)
                ref_latent, _, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts,
                                                      mask=mask_inpainting, index=index,
                                                      use_original_steps=use_original_steps,
                                                      unconditional_guidance_scale=1.0,
                                                      unconditional_conditioning=None)
                ref_latent = torch.clamp(ref_latent, -3.0, 3.0)

                if use_spatial_fusion:
                    # 使用平滑掩码
                    x_dec = self.spatial_blend(ref_latent, x_dec, mask_smooth, feather_radius)
                else:
                    ref_latent_dct = dct_2d(ref_latent, norm='ortho')
                    x_dec_dct = dct_2d(x_dec, norm='ortho')
                    merged_dct = low_pass(ref_latent_dct, threshold_val) + high_pass(x_dec_dct, threshold_val + 1)
                    x_dec = idct_2d(merged_dct, norm='ortho')
                x_dec = torch.clamp(x_dec, -3.0, 3.0)

                x_dec, _, _ = self.p_sample_ddim(x_dec, cond_with_text, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=7.5,
                                                 unconditional_conditioning=conds_with_text)
                x_dec = torch.clamp(x_dec, -3.0, 3.0)

                if use_spatial_fusion:
                    blend_mask = self.smooth_mask(mask_smooth, sigma=1.0) if feather_radius > 3 else mask_smooth
                    ref_latent1 = self.spatial_blend(ref_latent1, x_dec, blend_mask, feather_radius)
                else:
                    x_dec_dct = dct_2d(x_dec, norm='ortho')
                    ref_latent_dct1 = dct_2d(ref_latent1, norm='ortho')
                    if use_edge_guidance and self.edge_guider is not None:
                        target_size = ref_latent_dct1.shape[-2:]
                        edge_weight = self.edge_guider.get_edge_weight(ref_latent1, self.model, mask, target_size)
                        edge_weight = torch.clamp(edge_weight * edge_guidance_weight, 0.0, 1.0)
                        low_f = low_pass(ref_latent_dct1, threshold1)
                        high_f = high_pass(low_pass(x_dec_dct, threshold2), threshold1 + 1)
                        merged_dct = self.edge_guider.guided_fusion(low_f, high_f, edge_weight)
                        merged_dct = merged_dct + high_pass(ref_latent_dct1, threshold2 + 1)
                    else:
                        merged_dct = low_pass(ref_latent_dct1, threshold1) + \
                                     high_pass(low_pass(x_dec_dct, threshold2), threshold1 + 1) + \
                                     high_pass(ref_latent_dct1, threshold2 + 1)
                    ref_latent1 = idct_2d(merged_dct, norm='ortho')
                ref_latent1 = torch.clamp(ref_latent1, -3.0, 3.0)

                # 已改回 i % 3
                if use_adversarial and self.adversarial_guidance is not None and i % 3 == 0:
                    ref_latent1 = self.adversarial_guidance.guided_sampling_step(ref_latent1, mask, i, total_steps)
                if use_ssim_constraint and original_latent is not None and i % 3 == 0:
                    ref_latent1 = self.apply_ssim_constraint(ref_latent1, original_latent, mask, ssim_weight)
                if use_lpips_constraint and original_latent is not None and i % 3 == 0:
                    ref_latent1 = self.apply_lpips_constraint(ref_latent1, original_latent, mask, lpips_weight)

                ref_latent1, _, _ = self.p_sample_ddim(ref_latent1, conds1, ts, index=index,
                                                       use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=1.0,
                                                       unconditional_conditioning=None)
                ref_latent1 = torch.clamp(ref_latent1, -3.0, 3.0)

            else:
                ref_latent1 = self._get_intermediate_step(intermediates, intermediate_steps, ts + 1, mask, ref_latent1)

                # 已改回 i % 3
                if use_adversarial and self.adversarial_guidance is not None and i % 3 == 0:
                    ref_latent1 = self.adversarial_guidance.guided_sampling_step(ref_latent1, mask, i, total_steps)
                if use_ssim_constraint and original_latent is not None and i % 3 == 0:
                    ref_latent1 = self.apply_ssim_constraint(ref_latent1, original_latent, mask, ssim_weight)
                if use_lpips_constraint and original_latent is not None and i % 3 == 0:
                    ref_latent1 = self.apply_lpips_constraint(ref_latent1, original_latent, mask, lpips_weight)

                ref_latent1, _, _ = self.p_sample_ddim(ref_latent1, cond_with_text, ts, index=index,
                                                       use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=7.5,
                                                       unconditional_conditioning=conds_with_text)
                ref_latent1 = torch.clamp(ref_latent1, -3.0, 3.0)

            if use_text_guidance and text_embedding is not None and self.embedding_optimizer is not None:
                if progress > 0.3 and i - last_optimization_step >= 3:
                    try:
                        ref_latent1 = self.apply_embedding_optimization(ref_latent1, text_embedding, mask,
                                                                        original_latent, progress)
                        last_optimization_step = i
                    except Exception as e:
                        print(f"  ⚠ 嵌入优化失败: {e}")

            if callback:
                callback(i)

            # ========== [NEW] 保存中间 latent 和图像 ==========
            if (save_latents_dir is not None or save_images_dir is not None) and (i % save_every == 0):
                # 保存 latent
                if save_latents_dir is not None:
                    os.makedirs(save_latents_dir, exist_ok=True)
                    lat_path = os.path.join(save_latents_dir, f"step_{i:06d}.pt")
                    torch.save(ref_latent1.cpu(), lat_path)
                # 保存图像
                if save_images_dir is not None:
                    os.makedirs(save_images_dir, exist_ok=True)
                    with torch.no_grad():
                        img = self.model.decode_first_stage(ref_latent1)
                        img = torch.clamp((img + 1) / 2, 0.0, 1.0)
                    img_path = os.path.join(save_images_dir, f"step_{i:06d}.png")
                    vutils.save_image(img, img_path, normalize=False)
            # ==================================================

        if return_latent:
            return ref_latent1

        with torch.no_grad():
            final_image = self.model.decode_first_stage(ref_latent1)
            final_image = torch.clamp((final_image + 1) / 2, 0.0, 1.0)

        # 后处理修复链（保持不变）
        if use_white_region_repair:
            final_image = self.repair_white_regions(final_image, mask=mask,
                                                    brightness_thresh=white_region_brightness_thresh,
                                                    min_area=white_region_min_area,
                                                    max_area=white_region_max_area,
                                                    inpaint_radius=white_region_inpaint_radius)
        if use_overflow_fix:
            final_image = self.fix_overflow_pixels(final_image, mask=mask,
                                                   low_thresh=overflow_low_thresh,
                                                   high_thresh=overflow_high_thresh,
                                                   min_area=overflow_min_area,
                                                   max_area=overflow_max_area,
                                                   inpaint_radius=overflow_inpaint_radius)
        if use_repair:
            final_image = self.repair_artifacts(final_image, mask,
                                                black_thresh=repair_black_thresh,
                                                color_diff_thresh=repair_color_sensitivity,
                                                sat_thresh=repair_sat_thresh,
                                                max_artifact_area=repair_max_area,
                                                iterations=repair_iterations,
                                                use_inpaint=repair_use_inpaint)
        if use_white_blob_removal:
            final_image = self.remove_small_blobs(final_image,
                                                  brightness_thresh=white_blob_brightness_thresh,
                                                  max_area=white_blob_max_area,
                                                  blur_radius=white_blob_blur_radius,
                                                  strength=white_blob_strength)
        if use_exploding_suppress:
            final_image = self.suppress_exploding_pixels(final_image, mask=mask,
                                                         brightness_thresh=explode_brightness_thresh,
                                                         color_imbalance=explode_color_imbalance,
                                                         local_window=explode_local_window,
                                                         local_ratio=explode_local_ratio,
                                                         min_area=explode_min_area,
                                                         max_area=explode_max_area,
                                                         use_inpaint=True,
                                                         inpaint_radius=explode_inpaint_radius)
        if use_white_patch_repair:
            final_image = self.repair_white_patches(final_image, mask=mask,
                                                    brightness_thresh=white_patch_brightness_thresh,
                                                    saturation_thresh=white_patch_saturation_thresh,
                                                    min_area=white_patch_min_area,
                                                    max_area=white_patch_max_area,
                                                    inpaint_radius=white_patch_inpaint_radius)
        return final_image

    # ---------- 新增 encode 方法（覆盖父类，保存每一步加噪图像） ----------
    # ---------- 覆盖父类 encode，添加每一步保存图像和打印 ----------
    # ---------- 覆盖父类 encode，每一步保存图像并打印 ----------
    def encode(self, x0, cond, t_enc, save_images_dir=None, save_every=1):
        # 关键修改：不再翻转，使用递增时间步
        timesteps = self.ddim_timesteps[:t_enc]  # ✅ 递增序列
        x = x0.clone()
        intermediates = []
        intermediate_steps = []

        if save_images_dir is not None:
            os.makedirs(save_images_dir, exist_ok=True)

        record_interval = max(1, t_enc // 50)

        for i, t in enumerate(timesteps):
            ts = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            x = self.model.q_sample(x, ts)  # 加噪一步

            if (i % record_interval == 0) or (i == len(timesteps) - 1):
                intermediate_steps.append(t)
                intermediates.append(x.clone().detach().cpu())

            if save_images_dir is not None and i % save_every == 0:
                with torch.no_grad():
                    img = self.model.decode_first_stage(x)
                    img = torch.clamp((img + 1) / 2, 0.0, 1.0)
                    img_path = os.path.join(save_images_dir, f"encode_step_{i:06d}.png")
                    vutils.save_image(img, img_path, normalize=False)
                    print(f"[Encode] 已保存第 {i} 步图像 -> {img_path}")

        out = {
            'intermediate_steps': intermediate_steps,
            'intermediates': intermediates
        }
        return x, out