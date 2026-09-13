# SMR-Diff

**Title**: SMR-DIFF: MASK-AWARE SPATIAL MIXING AND DEFECT REPAIR FOR TEXT-GUIDED IMAGE INPAINTING

**Writer**:
* Xiaowei Duan¹
* Yao Li¹†
* Zhixiong Yang²*
* Yajun Zhang¹*

**Organization**:
* ¹ School of Software, Xinjiang University, Urumqi, China
* ² School of Computer Science and Technology, Xinjiang University, Urumqi, China
---
## Motivation

Text-guided image inpainting needs to coordinate two complementary priors: the reference image provides structural and appearance information, while the text prompt guides semantic generation. Existing methods often combine them through rigid spatial composition or frequency-domain replacement, which may cause boundary discontinuities or limited text controllability. This motivates us to formulate image inpainting as a spatial information allocation problem. SMR-Diff performs mask-aware spatial mixing during diffusion sampling, preserving reference information in unmasked regions while introducing text-conditioned semantics into masked regions and ensuring smooth mask-boundary transitions.



## Introduction

In this paper, we propose a Spatial Mixing and Repair Diffusion framework, dubbed **SMR-Diff**, for text-guided image inpainting, by formulating image inpainting as a spatial information allocation problem, where reference-image information and text-conditioned semantics are selectively allocated across spatial regions during the diffusion process, while preserving unmasked regions, to circumvent two challenges in a row. Specifically, a null-text reference branch extracts structural and appearance information from unmasked regions, while a text-guided branch provides semantic cues for masked-region generation. Their contributions are progressively regulated through mask-aware spatial mixing and reference latent optimization, followed by a null-text mask-driven refinement stage to enhance spatial consistency and reduce reference-content interference. Finally, a defect-aware post-processing module with Poisson blending further corrects residual artifacts and improves boundary continuity. Extensive experiments on BrushBench and EditBench validate the superiority of SMR-Diff over state-of-the-art diffusion models for text-guided image inpainting.




**Three-Stage Mask-Aware Spatial Mixing Diffusion Framework:**

**1. Null-Text Reference Denoising Process**

$$\hat{z}_{t}^{nrl} = \mathcal{D}_{\theta}^{MSA,NCA}(z_{t}^{nrl}, t, \tau_{\theta}(c_{\emptyset}); M)$$

$$z_{t}^{blend} = M_{t} \odot \hat{z}_{t}^{nrl} + (1 - M_{t}) \odot z_{t}^{ref}$$

**2. Text-Guided Global Spatial Denoising Process**

$$z_{t}^{text} = \mathcal{D}_{\theta}^{SA,TCA}(z_{t}^{blend}, t, \tau_{\theta}(c))$$

$$z_{t}^{mix} = (1 - M_{t}) \odot z_{t}^{nrl-1} + M_{t} \odot z_{t}^{text}$$

**3. Null-Text Mask-Driven Refinement Process**

$$z_{t}^{out} = \mathcal{D}_{\theta}^{SA,NCA}(z_{t}^{mix}, t, \tau_{\theta}(c_{\emptyset}); M_{t})$$



## Example Results

* Visual comparison between our method and the competitors.

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/eb975fb5-939b-4541-b313-788afd9fed73" />

* Quantitative results.
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/ae87ab65-634b-434f-9778-327bef6d64d2" />
<img width="622" height="165" alt="image" src="https://github.com/user-attachments/assets/6acd55af-c363-488b-81e2-4cde3f9da832" />


* Ablation Studies.
 <img width="1380" height="432" alt="image" src="https://github.com/user-attachments/assets/18d892da-ca72-42be-843d-9957522791da" />


## Inference

1. Dataset Preparation: [BrushBench](https://github.com/TencentARC/BrushNet) and  [EditBench](https://imagen.research.google/editor/)

2. Pre-trained models: [Realistic Vision V6.0 B1](https://civitai.com/models/4201/realistic-vision-v60-b1?modelVersionId=501240)

3. Run the following command:

```bash
Python testceshi.py
