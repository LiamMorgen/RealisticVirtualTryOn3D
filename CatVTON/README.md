# [ICLR 25]🐈 CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="http://arxiv.org/abs/2407.15886" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2407.15886-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/zhengchong/CatVTON' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="https://huggingface.co/spaces/zhengchong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href="https://huggingface.co/spaces/zhengchong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://zheng-chong.github.io/CatVTON/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON/LICENCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>


**CatVTON** is a simple and efficient virtual try-on diffusion model with ***1) Lightweight Network (899.06M parameters totally)***, ***2) Parameter-Efficient Training (49.57M parameters trainable)*** and ***3) Simplified Inference (< 8G VRAM for 1024X768 resolution)***.
<div align="center">
  <img src="resource/img/teaser.jpg" width="100%" height="100%"/>
</div>



## Updates 
- **`2025/02/24`**: 🎉 We are excited to announce [**CatV2TON**](https://github.com/Zheng-Chong/CatV2TON), our new DiT-based model that supports both **image and video try-on**! Check it out!
- **`2025/02/20`**: Our [**Paper on ArXiv**](http://arxiv.org/abs/2407.15886) has been updated to v2, which includes more details.
- **`2025/01/24`**: 🥳 CatVTON has been accepted to **ICLR 2025**!
- **`2024/12/20`**: 😄 Code for gradio app of **CatVTON-FLUX** has been released! It is not a stable version, but it is a good start!
- **`2024/12/19`**: [**CatVTON-FLUX**](https://huggingface.co/spaces/zhengchong/CatVTON) has been released! It is a extremely lightweight LoRA (only 37.4M checkpints) for [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), the lora weights are available in **[huggingface repo](https://huggingface.co/zhengchong/CatVTON/tree/main/flux-lora)**, code will be released soon!
- **`2024/11/26`**: Our **unified vision-based model for image and video try-on** will be released soon, bringing a brand-new virtual try-on experience! While our demo page will be temporarily taken offline, [**the demo on HuggingFace Space**](https://huggingface.co/spaces/zhengchong/CatVTON) will remain available for use !
- **`2024/10/17`**:[**Mask-free version**](https://huggingface.co/zhengchong/CatVTON-MaskFree)🤗 of CatVTON is release ! 
- **`2024/10/13`**: We have built a repo [**Awesome-Try-On-Models**](https://github.com/Zheng-Chong/Awesome-Try-On-Models) that focuses on image, video, and 3D-based try-on models published after 2023, aiming to provide insights into the latest technological trends. If you're interested, feel free to contribute or give it a 🌟 star!
- **`2024/08/13`**: We localize DensePose & SCHP to avoid certain environment issues.
- **`2024/08/10`**: Our 🤗 [**HuggingFace Space**](https://huggingface.co/spaces/zhengchong/CatVTON) is available now! Thanks for the grant from [**ZeroGPU**](https://huggingface.co/zero-gpu-explorers)！
- **`2024/08/09`**: [**Evaluation code**](https://github.com/Zheng-Chong/CatVTON?tab=readme-ov-file#3-calculate-metrics) is provided to calculate metrics 📚.
- **`2024/07/27`**: We provide code and workflow for deploying CatVTON on [**ComfyUI**](https://github.com/Zheng-Chong/CatVTON?tab=readme-ov-file#comfyui-workflow) 💥.
- **`2024/07/24`**: Our [**Paper on ArXiv**](http://arxiv.org/abs/2407.15886) is available 🥳!
- **`2024/07/22`**: Our [**App Code**](https://github.com/Zheng-Chong/CatVTON/blob/main/app.py) is released, deploy and enjoy CatVTON on your mechine 🎉!
- **`2024/07/21`**: Our [**Inference Code**](https://github.com/Zheng-Chong/CatVTON/blob/main/inference.py) and [**Weights** 🤗](https://huggingface.co/zhengchong/CatVTON) are released.
- **`2024/07/11`**: Our [**Online Demo**](https://huggingface.co/spaces/zhengchong/CatVTON) is released 😁.




## Installation 1

Create a conda environment & Install requirments
```shell
conda create -n catvton python==3.9.0
conda activate catvton
cd CatVTON-main  # or your path to CatVTON project dir
pip install -r requirements.txt
```

## Deployment 
### ComfyUI Workflow
We have modified the main code to enable easy deployment of CatVTON on [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Due to the incompatibility of the code structure, we have released this part in the [Releases](https://github.com/Zheng-Chong/CatVTON/releases/tag/ComfyUI), which includes the code placed under `custom_nodes` of ComfyUI and our workflow JSON files.

To deploy CatVTON to your ComfyUI, follow these steps:
1. Install all the requirements for both CatVTON and ComfyUI, refer to [Installation Guide for CatVTON](https://github.com/Zheng-Chong/CatVTON/blob/main/INSTALL.md) and [Installation Guide for ComfyUI](https://github.com/comfyanonymous/ComfyUI?tab=readme-ov-file#installing).
2. Download [`ComfyUI-CatVTON.zip`](https://github.com/Zheng-Chong/CatVTON/releases/download/ComfyUI/ComfyUI-CatVTON.zip) and unzip it in the `custom_nodes` folder under your ComfyUI project (clone from [ComfyUI](https://github.com/comfyanonymous/ComfyUI)).
3. Run the ComfyUI.
4. Download [`catvton_workflow.json`](https://github.com/Zheng-Chong/CatVTON/releases/download/ComfyUI/catvton_workflow.json) and drag it into you ComfyUI webpage and enjoy 😆!

> Problems under Windows OS, please refer to [issue#8](https://github.com/Zheng-Chong/CatVTON/issues/8).
> 
When you run the CatVTON workflow for the first time, the weight files will be automatically downloaded, usually taking dozens of minutes.

<div align="center">
  <img src="resource/img/comfyui-1.png" width="100%" height="100%"/>
</div>

<!-- <div align="center">
 <img src="resource/img/comfyui.png" width="100%" height="100%"/>
</div> -->

### Gradio App

To deploy the Gradio App for CatVTON on your machine, run the following command, and checkpoints will be automatically downloaded from HuggingFace.

```PowerShell
CUDA_VISIBLE_DEVICES=0 python app.py \
--output_dir="resource/demo/output" \
--mixed_precision="bf16" \
--allow_tf32 
```
When using `bf16` precision, generating results with a resolution of `1024x768` only requires about `8G` VRAM.

## Inference
### 1. Data Preparation
Before inference, you need to download the [VITON-HD](https://github.com/shadow2496/VITON-HD) or [DressCode](https://github.com/aimagelab/dress-code) dataset.
Once the datasets are downloaded, the folder structures should look like these:
```
├── VITON-HD
|   ├── test_pairs_unpaired.txt
│   ├── test
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── agnostic-mask
│   │   │   ├── [000006_00_mask.png | 000008_00.png | ...]
...
```

```
├── DressCode
|   ├── test_pairs_paired.txt
|   ├── test_pairs_unpaired.txt
│   ├── [dresses | lower_body | upper_body]
|   |   ├── test_pairs_paired.txt
|   |   ├── test_pairs_unpaired.txt
│   │   ├── images
│   │   │   ├── [013563_0.jpg | 013563_1.jpg | 013564_0.jpg | 013564_1.jpg | ...]
│   │   ├── agnostic_masks
│   │   │   ├── [013563_0.png| 013564_0.png | ...]
...
```
For the DressCode dataset, we provide script to preprocessed agnostic masks, run the following command:
```PowerShell
CUDA_VISIBLE_DEVICES=0 python preprocess_agnostic_mask.py \
--data_root_path <your_path_to_DressCode> 
```

### 2. Inference on VTIONHD/DressCode
To run the inference on the DressCode or VITON-HD dataset, run the following command, checkpoints will be automatically downloaded from HuggingFace.

```PowerShell
CUDA_VISIBLE_DEVICES=0 python inference.py \
--dataset [dresscode | vitonhd] \
--data_root_path <path> \
--output_dir <path> 
--dataloader_num_workers 8 \
--batch_size 8 \
--seed 555 \
--mixed_precision [no | fp16 | bf16] \
--allow_tf32 \
--repaint \
--eval_pair  
```
### 3. Calculate Metrics

After obtaining the inference results, calculate the metrics using the following command: 

```PowerShell
CUDA_VISIBLE_DEVICES=0 python eval.py \
--gt_folder <your_path_to_gt_image_folder> \
--pred_folder <your_path_to_predicted_image_folder> \
--paired \
--batch_size=16 \
--num_workers=16 
```

-  `--gt_folder` and `--pred_folder` should be folders that contain **only images**.
- To evaluate the results in a paired setting, use `--paired`; for an unpaired setting, simply omit it.
- `--batch_size` and `--num_workers` should be adjusted based on your machine.


## Acknowledgement
Our code is modified based on [Diffusers](https://github.com/huggingface/diffusers). We adopt [Stable Diffusion v1.5 inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) as the base model. We use [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing/tree/master) and [DensePose](https://github.com/facebookresearch/DensePose) to automatically generate masks in our [Gradio](https://github.com/gradio-app/gradio) App and [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflow. Thanks to all the contributors!

## License
All the materials, including code, checkpoints, and demo, are made available under the [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. You are free to copy, redistribute, remix, transform, and build upon the project for non-commercial purposes, as long as you give appropriate credit and distribute your contributions under the same license.


## Citation

```bibtex
@misc{chong2024catvtonconcatenationneedvirtual,
 title={CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models}, 
 author={Zheng Chong and Xiao Dong and Haoxiang Li and Shiyue Zhang and Wenqing Zhang and Xujie Zhang and Hanqing Zhao and Xiaodan Liang},
 year={2024},
 eprint={2407.15886},
 archivePrefix={arXiv},
 primaryClass={cs.CV},
 url={https://arxiv.org/abs/2407.15886}, 
}
```
