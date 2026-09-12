# SMR-Diff

**论文标题**: SMR-DIFF: MASK-AWARE SPATIAL MIXING AND DEFECT REPAIR FOR TEXT-GUIDED IMAGE INPAINTING

**作者**:
* Xiaowei Duan¹
* Yao Li¹†
* Zhixiong Yang²*
* Yajun Zhang¹*

**单位**:
* ¹ School of Software, Xinjiang University, Urumqi, China
* ² School of Computer Science and Technology, Xinjiang University, Urumqi, China

*注：† 同等贡献；* 通讯作者（Zhixiong Yang, Yajun Zhang）

---

## 动机

文本引导图像修复需要协调两个互补的先验信息：参考图像提供结构和外观信息，而文本提示引导语义生成。现有的方法通常通过僵硬的空间组合或频域替换将它们结合起来，这可能会导致边界不连续或文本可控性有限。这促使我们将图像修复公式化为一个空间信息分配问题。SMR-Diff 在扩散采样过程中执行掩码感知的空间混合，在保留未掩码区域参考信息的同时，将文本条件语义引入掩码区域，并确保平滑的掩码边界过渡。
```[cite: 2]

---

### 2. 英文 Markdown 代码块形式

```markdown
# SMR-Diff

**Title**: SMR-DIFF: MASK-AWARE SPATIAL MIXING AND DEFECT REPAIR FOR TEXT-GUIDED IMAGE INPAINTING

**Authors**:
* Xiaowei Duan¹
* Yao Li¹†
* Zhixiong Yang²*
* Yajun Zhang¹*

**Affiliations**:
* ¹ School of Software, Xinjiang University, Urumqi, China
* ² School of Computer Science and Technology, Xinjiang University, Urumqi, China

*Notes: † Equal contribution; * Corresponding authors (Zhixiong Yang, Yajun Zhang)

---

## Motivation

Text-guided image inpainting needs to coordinate two complementary priors: the reference image provides structural and appearance information, while the text prompt guides semantic generation. Existing methods often combine them through rigid spatial composition or frequency-domain replacement, which may cause boundary discontinuities or limited text controllability. This motivates us to formulate image inpainting as a spatial information allocation problem. SMR-Diff performs mask-aware spatial mixing during diffusion sampling, preserving reference information in unmasked regions while introducing text-conditioned semantics into masked regions and ensuring smooth mask-boundary transitions.




























## Inference

1. Dataset Preparation: [BrushBench](https://github.com/TencentARC/BrushNet)

2. Pre-trained models: [Realistic Vision V6.0 B1](https://civitai.com/models/4201/realistic-vision-v60-b1?modelVersionId=501240)

3. Run the following command:

```bash
Python test.py
