import torch
import torch.nn.functional as F
from tqdm import tqdm


class EmbeddingSpaceOptimizer:
    """
    文本-图像联合嵌入空间优化器
    在解码过程中直接优化 latent，使其 CLIP 特征接近文本特征
    """

    def __init__(self, clip_model, device, steps=5, lr=0.005):
        """
        参数:
            clip_model: CLIP 模型
            device: 设备
            steps: 优化步数
            lr: 学习率
        """
        self.clip_model = clip_model
        self.device = device
        self.steps = steps
        self.lr = lr

        # 预计算 CLIP 图像预处理
        import clip
        self.preprocess = clip.load("ViT-B/32", device=device)[1]

    def get_clip_features(self, image):
        """
        获取图像的 CLIP 特征

        参数:
            image: 图像张量 [B, C, H, W]，值范围 [0, 1]
        返回:
            features: CLIP 特征 [B, 512]
        """
        with torch.no_grad():
            # 确保图像在 [0, 1] 范围
            if image.min() < 0:
                image = (image + 1) / 2

            # 调整到 CLIP 输入尺寸 (224x224)
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

            # 归一化到 CLIP 标准范围
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(image.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image.device)
            image_norm = (image_resized - mean) / std

            # 编码图像
            features = self.clip_model.encode_image(image_norm)
            features = features / features.norm(dim=-1, keepdim=True)

        return features

    def optimize_latent(self, latent, text_embedding, model, mask=None, original_latent=None):
        """
        优化 latent 使其 CLIP 特征接近文本特征

        参数:
            latent: 初始 latent [B, C, H, W]
            text_embedding: 文本嵌入 [B, D]
            model: 扩散模型
            mask: 修复掩码 [B, 1, H, W]，只优化 mask 区域
            original_latent: 原始 latent，用于保护已知区域
        返回:
            optimized_latent: 优化后的 latent
        """
        # 克隆并启用梯度
        latent_opt = latent.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latent_opt], lr=self.lr)

        # 文本特征归一化
        text_embedding = F.normalize(text_embedding, dim=-1)

        loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad()

            with torch.enable_grad():
                # 解码当前 latent
                image = model.decode_first_stage(latent_opt)

                # 确保图像在 [0, 1] 范围
                image_normalized = (image + 1) / 2
                image_normalized = torch.clamp(image_normalized, 0, 1)

                # 获取图像 CLIP 特征
                image_features = self.get_clip_features(image_normalized)

                # 计算余弦相似度损失（希望最大化相似度）
                # 损失 = 1 - 余弦相似度
                similarity = (image_features * text_embedding).sum(dim=-1)
                clip_loss = (1 - similarity).mean()

                # 如果有 mask，只计算 mask 区域的损失
                if mask is not None:
                    # 下采样 mask 到 latent 尺寸
                    mask_resized = F.interpolate(mask, size=latent_opt.shape[-2:], mode='nearest')

                    # 计算 mask 区域的梯度权重
                    grad_weight = mask_resized

                    # 额外：mask 区域外保持与原图一致
                    if original_latent is not None:
                        consistency_loss = F.mse_loss(latent_opt * (1 - mask_resized),
                                                      original_latent * (1 - mask_resized))
                        clip_loss = clip_loss + 0.1 * consistency_loss

                # 反向传播
                clip_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([latent_opt], max_norm=1.0)

            optimizer.step()
            loss_history.append(clip_loss.item())

            # 如果有 mask，保护非修复区域
            if mask is not None and original_latent is not None:
                with torch.no_grad():
                    mask_resized = F.interpolate(mask, size=latent_opt.shape[-2:], mode='nearest')
                    latent_opt = latent_opt * mask_resized + original_latent * (1 - mask_resized)

        # 打印优化信息
        if len(loss_history) > 0:
            print(f"    [嵌入优化] 损失: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")

        return latent_opt.detach()


class ProgressiveEmbeddingOptimizer(EmbeddingSpaceOptimizer):
    """
    渐进式嵌入空间优化器
    在解码的不同阶段进行优化，逐步细化
    """

    def __init__(self, clip_model, device, steps_per_stage=3, lr=0.005):
        super().__init__(clip_model, device, steps=steps_per_stage, lr=lr)
        self.steps_per_stage = steps_per_stage

        # 不同阶段的权重
        self.stage_weights = {
            'early': 0.3,  # 早期：弱引导，给模型更多自由度
            'middle': 0.6,  # 中期：中等引导
            'late': 1.0  # 后期：强引导，精细对齐
        }

    def optimize_at_stage(self, latent, text_embedding, model, mask, original_latent, progress):
        """
        根据解码进度进行优化

        参数:
            progress: 解码进度 (0-1)
        """
        # 根据进度选择权重
        if progress < 0.3:
            stage = 'early'
        elif progress < 0.7:
            stage = 'middle'
        else:
            stage = 'late'

        weight = self.stage_weights[stage]

        # 动态调整步数和学习率
        steps = max(1, int(self.steps_per_stage * weight))
        lr = self.lr * weight

        # 临时调整参数
        original_steps = self.steps
        original_lr = self.lr

        self.steps = steps
        self.lr = lr

        # 执行优化
        optimized = self.optimize_latent(latent, text_embedding, model, mask, original_latent)

        # 恢复参数
        self.steps = original_steps
        self.lr = original_lr

        return optimized