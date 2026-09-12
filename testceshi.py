import json
import numpy as np
import torch
from PIL import Image
import os
from pytorch_lightning import seed_everything
from ntn.tools import create_model, load_state_dict
from ntn.ntn_sampler import Ntn_Sampler
import torchvision.utils as vutils
import torch.nn.functional as F
from safetensors.torch import load_file
import cv2

# 导入预训练判别器
from vgg_discriminator import get_pretrained_discriminator

# ==================== 配置区（所有功能开关和参数保持不变） ====================
USE_ADVERSARIAL = False
USE_EDGE_GUIDANCE = True
USE_SSIM_CONSTRAINT = True
USE_TEXT_GUIDANCE = False

USE_WHITE_REGION_REPAIR = True
WHITE_REGION_BRIGHTNESS_THRESH = 0.90
WHITE_REGION_MIN_AREA = 200
WHITE_REGION_MAX_AREA = 20000
WHITE_REGION_INPAINT_RADIUS = 5

USE_EXPLODING_SUPPRESS = True
EXPLODE_BRIGHTNESS_THRESH = 0.92
EXPLODE_COLOR_IMBALANCE = 0.6
EXPLODE_LOCAL_WINDOW = 5
EXPLODE_LOCAL_RATIO = 0.35
EXPLODE_MIN_AREA = 2
EXPLODE_MAX_AREA = 200
EXPLODE_INPAINT_RADIUS = 3

USE_WHITE_BLOB_REMOVAL = False
USE_REPAIR = False
USE_OVERFLOW_FIX = False
RETURN_LATENT = False

ENCODE_STEPS = 1000
DECODE_STEPS = 100
LAMBDA_END = 0.6
UNCONDITIONAL_GUIDANCE_SCALE = 7.5
H = W = 512
num_samples = 1
# ============================================================================

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())
print("Selected Device:", device)


# ==================== 工具函数（不变） ====================
def preprocess_mask(maskt, latent):
    if isinstance(maskt, np.ndarray):
        maskt = torch.from_numpy(maskt)
    maskt = maskt.float()
    if maskt.dim() == 2:
        maskt = maskt.unsqueeze(0).unsqueeze(0)
    elif maskt.dim() == 3:
        maskt = maskt.unsqueeze(0)
    device = latent.device
    target_size = latent.shape[-2:]
    mask_resized = F.interpolate(maskt, size=target_size, mode='bilinear', align_corners=True)
    mask_resized = mask_resized.to(device)
    return torch.clamp(mask_resized, 0.0, 1.0)


def get_text_embedding(text, device):
    try:
        import clip
        clip_model, _ = clip.load("ViT-B/32", device=device)
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding
    except Exception as e:
        print(f"⚠ CLIP加载失败: {e}")
        return None


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ==================== 核心生成函数（修改：加噪每10步，去噪每一步） ====================
def generate_inpainting(
        img_tensor, mask_tensor, cond, un_cond, conds, conds1,
        text_embedding, use_text_guidance,
        save_intermediate_dir=None
):
    # 创建保存子目录
    if save_intermediate_dir is not None:
        os.makedirs(save_intermediate_dir, exist_ok=True)
        encode_dir = os.path.join(save_intermediate_dir, "encode")
        decode_dir = os.path.join(save_intermediate_dir, "decode")
        os.makedirs(encode_dir, exist_ok=True)
        os.makedirs(decode_dir, exist_ok=True)
    else:
        encode_dir = decode_dir = None

    sampler.make_schedule(ddim_num_steps=ENCODE_STEPS)
    encoder_posterior = model.encode_first_stage(img_tensor)
    z = model.get_first_stage_encoding(encoder_posterior).detach()

    # 调用 encode，每10步保存
    latent, out = sampler.encode(
        x0=z,
        cond=un_cond,
        t_enc=ENCODE_STEPS,
        save_images_dir=encode_dir,
        save_every=10
    )

    sampler.make_schedule(ddim_num_steps=DECODE_STEPS)
    mask_resized = preprocess_mask(mask_tensor, torch.randn(1, 4, 64, 64).cuda()).cuda()
    end_step = ENCODE_STEPS * LAMBDA_END

    # 调用 decode，每一步都保存
    x_rec = sampler.decode(
        ref_latent=latent,
        cond=cond,
        t_dec=DECODE_STEPS,
        unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
        unconditional_conditioning=un_cond,
        mask=mask_resized,
        unmask=out,
        conds=conds,
        conds1=conds1,
        threshold=-1,
        end_step=end_step,
        use_adversarial=USE_ADVERSARIAL,
        use_edge_guidance=USE_EDGE_GUIDANCE,
        use_ssim_constraint=USE_SSIM_CONSTRAINT,
        original_latent=z,
        text_embedding=text_embedding,
        use_text_guidance=use_text_guidance,
        use_white_region_repair=USE_WHITE_REGION_REPAIR,
        white_region_brightness_thresh=WHITE_REGION_BRIGHTNESS_THRESH,
        white_region_min_area=WHITE_REGION_MIN_AREA,
        white_region_max_area=WHITE_REGION_MAX_AREA,
        white_region_inpaint_radius=WHITE_REGION_INPAINT_RADIUS,
        use_exploding_suppress=USE_EXPLODING_SUPPRESS,
        explode_brightness_thresh=EXPLODE_BRIGHTNESS_THRESH,
        explode_color_imbalance=EXPLODE_COLOR_IMBALANCE,
        explode_local_window=EXPLODE_LOCAL_WINDOW,
        explode_local_ratio=EXPLODE_LOCAL_RATIO,
        explode_min_area=EXPLODE_MIN_AREA,
        explode_max_area=EXPLODE_MAX_AREA,
        explode_inpaint_radius=EXPLODE_INPAINT_RADIUS,
        use_white_blob_removal=USE_WHITE_BLOB_REMOVAL,
        use_repair=USE_REPAIR,
        use_overflow_fix=USE_OVERFLOW_FIX,
        return_latent=RETURN_LATENT,
        # 新增 decode 保存参数，每一步保存
        save_images_dir=decode_dir,
        save_every=1
    )
    return x_rec


# ==================== 单张处理函数（增加中间目录参数） ====================
def process_one_image(img_path, mask_path, caption, output_poisson_path, seed=42, intermediate_dir=None):
    seed_everything(seed)

    text_embedding = get_text_embedding(caption, device) if USE_TEXT_GUIDANCE else None
    actual_use_text = USE_TEXT_GUIDANCE and (text_embedding is not None)

    mask = np.array(Image.open(mask_path).resize((H, W)).convert('L'))
    mask = (mask.astype(np.float32) / 255.0)
    mask = 1.0 - mask
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()

    img = np.array(Image.open(img_path).resize((H, W)))[..., :3]
    img = (img.astype(np.float32) / 127.5) - 1.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].float().cuda()
    original_image = (img_tensor + 1) / 2

    un_cond = {"c_crossattn": [model.get_learned_conditioning(
        ['RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3'] * num_samples)]}
    cond = {"c_crossattn": [model.get_learned_conditioning([caption] * num_samples)]}
    conds = {"c_crossattn": [model.get_learned_conditioning(
        ['text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated'] * num_samples)]}
    conds1 = {"c_crossattn": [model.get_learned_conditioning([''] * num_samples)]}

    print("🚀 开始生成...")
    final_pred = generate_inpainting(
        img_tensor, mask_tensor, cond, un_cond, conds, conds1,
        text_embedding, actual_use_text,
        save_intermediate_dir=intermediate_dir
    )

    print("🔗 执行泊松融合...")
    gen_np = (final_pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    orig_np = (original_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask_np = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    center = (W // 2, H // 2)
    result = cv2.seamlessClone(gen_np, orig_np, mask_np, center, cv2.NORMAL_CLONE)
    final_image = torch.from_numpy(result.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

    vutils.save_image(final_image, output_poisson_path, normalize=False)

    psnr = calculate_psnr(final_image, original_image)
    print(f"📈 PSNR = {psnr:.2f} dB")
    try:
        from skimage.metrics import structural_similarity as ssim
        comp_gray = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
        ssim_val = ssim(comp_gray, orig_gray, data_range=255)
        print(f"📈 SSIM = {ssim_val:.4f}")
    except:
        pass


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n加载模型...")
    model = create_model('./models/model_ldm_v15.yaml').cuda()
    state_dict = load_file('./models/Realistic_Vision_V6.0_NV_B1.safetensors')
    model.load_state_dict(state_dict, strict=False)

    sampler = Ntn_Sampler(model)
    discriminator = get_pretrained_discriminator(device)
    sampler.init_adversarial_guidance(discriminator, guidance_weight=0.5)
    sampler.init_edge_guidance()

    print("✅ 模型初始化完成，开始单张生成！\n")

    # ========== 直接指定图片、掩码、提示词和输出路径 ==========
    img_path = "./data/images/000000565.jpg"
    mask_path = "./data/mask1/000000565_composite.png"
    caption = "a close up of a cat looking to the side"
    output_poisson = "./data/000000565_poisson.png"
    SEED = 42

    # 指定中间结果保存目录
    intermediate_dir = "./intermediate_results"  # 若不想保存设为 None
    # =======================================================

    print(f"\n{'=' * 60}")
    print(f"单张处理 | 图片: {img_path}")
    print(f"掩码: {mask_path}")
    print(f"提示词: {caption}")
    print(f"中间结果保存至: {intermediate_dir if intermediate_dir else '（不保存）'}")
    print(f"{'=' * 60}")

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"⚠️ 文件缺失，请检查路径")
    else:
        try:
            process_one_image(img_path, mask_path, caption, output_poisson, SEED, intermediate_dir)
            print("\n🎉 单张图片处理完成！")
        except Exception as e:
            print(f"❌ 处理失败: {e}")