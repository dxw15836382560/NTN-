# vgg_discriminator.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VGG16FeatureExtractor(nn.Module):
    """VGG16特征提取器（预训练在ImageNet上）"""

    def __init__(self):
        super().__init__()
        # 加载预训练VGG16
        vgg16 = models.vgg16(pretrained=True)

        # 使用特征提取部分（不含分类头）
        self.features = vgg16.features

        # 冻结所有参数，不参与训练
        for param in self.features.parameters():
            param.requires_grad = False

        # 定义要使用的特征层
        self.layers = [3, 8, 15, 22]  # conv1_2, conv2_2, conv3_3, conv4_3

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features


class PerceptualDiscriminator(nn.Module):
    """
    基于感知特征的判别器 - 加强版
    使用特征方差作为真实性分数，产生更强的梯度
    """

    def __init__(self, use_variance=True):
        super().__init__()
        self.feature_extractor = VGG16FeatureExtractor()
        self.use_variance = use_variance

        # 多层特征的权重
        self.layer_weights = [0.1, 0.3, 0.4, 0.2]

        # 可选：一个简单的卷积头增强特征
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
        )

        # 移动到GPU后设置为评估模式
        self.eval()

    def forward(self, x):
        # 确保输入在正确范围内
        if x.min() >= 0 and x.max() <= 1:
            # 归一化到ImageNet标准
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std

        # 提取多尺度特征
        features = self.feature_extractor(x)

        if self.use_variance:
            # 使用特征方差作为真实性分数（产生更强的梯度信号）
            scores = []
            for i, feat in enumerate(features):
                # 计算每个通道的方差
                variance = torch.var(feat, dim=[2, 3], keepdim=True)
                # 取所有通道的平均
                score = variance.mean(dim=1, keepdim=True)
                # 应用层权重
                scores.append(score * self.layer_weights[i])

            # 融合多尺度分数
            final_score = sum(scores)

            # 可选：增强特征
            last_feat = features[-1]
            enhanced = self.enhance_conv(last_feat)
            # 计算增强特征的方差
            enhanced_variance = torch.var(enhanced, dim=[2, 3], keepdim=True).mean(dim=1, keepdim=True)
            final_score = final_score + 0.5 * enhanced_variance

            # 归一化到合理范围
            final_score = torch.sigmoid(final_score * 0.1)

            return final_score
        else:
            # 原始方法
            last_feat = features[-1]
            score = self.enhance_conv(last_feat)
            score = torch.var(score, dim=[2, 3], keepdim=True).mean(dim=1, keepdim=True)
            return torch.sigmoid(score * 0.1)


def get_pretrained_discriminator(device='cuda'):
    """
    获取预训练判别器的便捷函数
    """
    print("正在加载预训练VGG16判别器（加强版）...")
    discriminator = PerceptualDiscriminator(use_variance=True)
    discriminator = discriminator.to(device)
    discriminator.eval()
    print("✓ 判别器加载完成！（基于ImageNet预训练，使用特征方差）")
    return discriminator


# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = get_pretrained_discriminator(device)

    # 测试不同图像的输出
    test_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        score = discriminator(test_input)
        print(f"随机噪声 score: {score.mean().item():.4f}")

    # 测试真实图像风格的输入（假设更真实）
    test_real = torch.ones(1, 3, 512, 512).to(device) * 0.5
    with torch.no_grad():
        score = discriminator(test_real)
        print(f"平滑图像 score: {score.mean().item():.4f}")

    print("✓ 判别器测试通过！")