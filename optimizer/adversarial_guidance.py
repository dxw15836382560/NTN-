import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast


class AdversarialGuidance:
    """对抗性引导采样器"""

    def __init__(self, discriminator, model, guidance_weight=0.3):
        self.discriminator = discriminator
        self.model = model
        self.guidance_weight = guidance_weight
        self.gradient_stats = []  # 记录梯度统计信息
        self.call_count = 0

    def compute_adversarial_gradient(self, latent, mask=None):
        """计算对抗性梯度"""
        with torch.enable_grad():
            # 需要梯度
            latent_with_grad = latent.clone().detach().requires_grad_(True)

            # 解码到图像空间 - 使用 enable_grad 确保梯度流通
            try:
                with torch.enable_grad():
                    image = self.model.decode_first_stage(latent_with_grad)
            except Exception as e:
                print(f"  ⚠ Error in decode_first_stage: {e}")
                return torch.zeros_like(latent)

            # 计算判别器分数
            scores = self.discriminator(image)

            # 处理多尺度判别器的输出
            if isinstance(scores, (list, tuple)):
                # 如果是多尺度判别器，取平均
                total_loss = 0
                valid_scores = 0
                for score in scores:
                    if score is not None:
                        if mask is not None:
                            # 调整mask到与score相同的尺寸
                            if mask.shape[-2:] != score.shape[-2:]:
                                mask_resized = F.interpolate(mask, size=score.shape[-2:],
                                                             mode='bilinear', align_corners=True)
                            else:
                                mask_resized = mask
                            # 计算损失
                            masked_score = score * mask_resized
                            loss = -torch.mean(masked_score)
                        else:
                            loss = -torch.mean(score)
                        total_loss = total_loss + loss
                        valid_scores += 1
                if valid_scores > 0:
                    loss = total_loss / valid_scores
                else:
                    return torch.zeros_like(latent)
            else:
                # 单尺度判别器
                if mask is not None:
                    # 调整mask到与score相同的尺寸
                    if mask.shape[-2:] != scores.shape[-2:]:
                        mask_resized = F.interpolate(mask, size=scores.shape[-2:],
                                                     mode='bilinear', align_corners=True)
                    else:
                        mask_resized = mask
                    masked_score = scores * mask_resized
                    loss = -torch.mean(masked_score)
                else:
                    loss = -torch.mean(scores)

            # 计算梯度，添加 allow_unused=True 避免报错
            gradient = torch.autograd.grad(loss, latent_with_grad,
                                           allow_unused=True,
                                           retain_graph=False)[0]

            # 如果梯度为 None，返回零梯度
            if gradient is None:
                print(f"  ⚠ WARNING: Gradient is None, returning zero gradient")
                return torch.zeros_like(latent)

            # 对梯度进行归一化，避免梯度爆炸或消失
            gradient_norm = gradient.norm()
            if gradient_norm > 0:
                gradient = gradient / (gradient_norm + 1e-8)

        return gradient

    def compute_feature_matching_loss(self, real_features, fake_features):
        """特征匹配损失 - 对齐特征统计"""
        loss = 0
        if isinstance(real_features, (list, tuple)) and isinstance(fake_features, (list, tuple)):
            for real_feat, fake_feat in zip(real_features, fake_features):
                loss += F.l1_loss(real_feat.mean(dim=[2, 3]), fake_feat.mean(dim=[2, 3]))
        else:
            loss = F.l1_loss(real_features.mean(dim=[2, 3]), fake_features.mean(dim=[2, 3]))
        return loss

    def adaptive_guidance(self, latent, mask, step, total_steps):
        """自适应引导强度 - 增强版"""
        # 根据步数调整
        progress = step / total_steps

        if mask is not None:
            mask_ratio = mask.mean().item()
            # 大面积修复需要更强的引导
            if mask_ratio > 0.5:
                # 增强基础强度
                strength = self.guidance_weight * (0.8 + progress * 0.5)
            else:
                # 增强基础强度
                strength = self.guidance_weight * (0.5 + progress * 0.5)
        else:
            strength = self.guidance_weight * (0.5 + progress * 0.5)

        return strength

    def guided_sampling_step(self, latent, mask, step, total_steps):
        """单步引导采样（带调试信息）"""
        self.call_count += 1
        try:
            # 计算对抗性梯度
            gradient = self.compute_adversarial_gradient(latent, mask)

            # ========== 调试信息 ==========
            gradient_norm = gradient.norm().item()
            gradient_mean = gradient.mean().item()
            gradient_std = gradient.std().item()

            # 每5步打印一次，避免输出太多
            if step % 5 == 0 or step == total_steps - 1:
                print(f"[Step {step:3d}/{total_steps}] "
                      f"Grad norm: {gradient_norm:.6f}, "
                      f"Mean: {gradient_mean:.6f}, "
                      f"Std: {gradient_std:.6f}, "
                      f"Weight: {self.guidance_weight:.3f}")

                if gradient_norm < 1e-6:
                    print(f"  ⚠ WARNING: Gradient too small! Guidance may be ineffective.")
                elif gradient_norm > 1e-2:
                    print(f"  ⚠ WARNING: Gradient too large! May destabilize sampling.")
                else:
                    print(f"  ✓ Gradient OK - Guidance should be effective.")
            # ========== 调试信息结束 ==========

            # 检查梯度是否有效
            if torch.isnan(gradient).any() or torch.isinf(gradient).any():
                print(f"  ⚠ WARNING: Invalid gradient (NaN/Inf) at step {step}, skipping guidance")
                return latent

            # 自适应强度
            strength = self.adaptive_guidance(latent, mask, step, total_steps)

            # 应用梯度更新
            guided_latent = latent + strength * gradient

            return guided_latent

        except Exception as e:
            print(f"  ⚠ ERROR: Adversarial guidance failed at step {step}: {e}")
            return latent