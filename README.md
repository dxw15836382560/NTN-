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
---
## Motivation

Text-guided image inpainting needs to coordinate two complementary priors: the reference image provides structural and appearance information, while the text prompt guides semantic generation. Existing methods often combine them through rigid spatial composition or frequency-domain replacement, which may cause boundary discontinuities or limited text controllability. This motivates us to formulate image inpainting as a spatial information allocation problem. SMR-Diff performs mask-aware spatial mixing during diffusion sampling, preserving reference information in unmasked regions while introducing text-conditioned semantics into masked regions and ensuring smooth mask-boundary transitions.
[fig_1v2 (1).pdf](https://github.com/user-attachments/files/32145681/fig_1v2.1.pdf)



























## Inference

1. Dataset Preparation: [BrushBench](https://github.com/TencentARC/BrushNet)

2. Pre-trained models: [Realistic Vision V6.0 B1](https://civitai.com/models/4201/realistic-vision-v60-b1?modelVersionId=501240)

3. Run the following command:

```bash
Python test.py
