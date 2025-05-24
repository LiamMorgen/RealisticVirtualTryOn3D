import argparse
import os
from datetime import datetime
import subprocess  # 用于执行终端命令
import glob  # 用于文件查找
import time  # 用于等待文件生成
import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import shutil

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    # 创建与图像同名的文件夹及子文件夹
    image_name = date_str[8:]
    folder_path = os.path.join(tmp_folder, date_str[:8], image_name)
    os.makedirs(folder_path, exist_ok=True)

    # 创建与图像同名的文件夹及子文件夹
    image_name = date_str[8:]
    folder_path = os.path.join(tmp_folder, date_str[:8], image_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # 创建软链接到SiTH目录
    target_link = "../SiTH/data/From_CatVTON"
    
    # 确保使用绝对路径
    folder_path_abs = os.path.abspath(folder_path)
    
    # 创建或更新软链接
    if os.path.islink(target_link):
        os.unlink(target_link)  # 如果软链接已存在，先删除
    os.symlink(folder_path_abs, target_link)  # 使用绝对路径创建新的软链接
    
    # 创建子文件夹
    subfolders = ["rgba", "smplx", "meshes", "images", "back_images"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(folder_path, subfolder), exist_ok=True)

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    # 这里需要修改cloth_type映射，因为UI中改为了中文
    cloth_type_map = {"上衣": "upper", "下装": "lower", "全身": "overall"}
    automasker_cloth_type = cloth_type_map.get(cloth_type, "upper") # 使用新变量存储映射后的类型

    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            automasker_cloth_type # 使用映射后的英文类型
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    # try:
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    # except Exception as e:
    #     raise gr.Error(
    #         "An error occurred. Please try again later: {}".format(e)
    #     )
    
    # Post-process
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    
    # 复制图像到rgba子文件夹 - 只保存结果图像
    images_subfolder_path = os.path.join(folder_path, "rgba")
    result_image.save(os.path.join(images_subfolder_path, image_name + ".png"))
    
    # 执行3D模型生成命令
    model_path = None
    try:
        # 保存当前工作目录
        current_dir = os.getcwd()
        
        # 切换到SiTH目录并执行脚本
        sith_dir = "../SiTH" # 改为同级文件夹下SiTH的项目路径
        os.chdir(sith_dir)
        print("执行3D模型生成命令...，约50s...")
        subprocess.run(['bash', 'run_work_test_examples.sh'], check=True)
        
        # 等待3D模型生成完成
        print("3D模型生成完成")
        # 可以适当增加等待时间，如果模型生成较慢
        time.sleep(7) 
        
        # 返回工作目录
        os.chdir(current_dir)

        # 直接在目标输出目录查找生成的3D模型文件
        model_dir = os.path.join(folder_path, "meshes")
        model_files = glob.glob(os.path.join(model_dir, "*.obj"))
        
        # 获取最新生成的3D模型文件
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)
            print(f"找到3D模型: {model_path}")
        else:
            # 如果仍然找不到，可以打印更详细的调试信息
            print(f"在目标目录 {model_dir} 未找到 .obj 文件。请检查 run_work_test_examples.sh 脚本的输出和权限。")
            # 可以在这里添加对 SiTH 日志文件的检查

    except Exception as e:
        print(f"执行3D模型生成或查找时出错: {str(e)}")
        # 确保返回工作目录
        if 'current_dir' in locals() and os.getcwd() != current_dir:
             os.chdir(current_dir)

    # 这里不再需要重复映射 cloth_type
    # cloth_type_map = {"上衣": "upper", "下装": "lower", "全身": "overall"}
    # cloth_type = cloth_type_map.get(cloth_type, "upper")
    
    # 这里需要修改show_type映射
    if show_type == "仅结果":
        return_image = result_image
    else:
        width, height = person_image.size
        if show_type == "输入和结果":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
        return_image = new_result_image
    
    return return_image, model_path


def person_example_fn(image_path):
    return image_path

HEADER = """
<h1 style="text-align: center;">逼真着装数字人生成系统设计与实现</h1>
<div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
  <table style="border-collapse: collapse; width: 50%;">
    <tr>
      <td style="padding: 8px; text-align: right;">学生姓名：</td>
      <td style="padding: 8px; text-align: left;">张密</td>
      <td style="padding: 8px; text-align: right;">学  号：</td>
      <td style="padding: 8px; text-align: left;">2021211806</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: right;">指导教师：</td>
      <td style="padding: 8px; text-align: left;">徐宗懿</td>
      <td style="padding: 8px; text-align: right;">所在单位：</td>
      <td style="padding: 8px; text-align: left;">计算机科学与技术学院\\人工智能学院</td>
    </tr>
  </table>
</div>
"""

def app_gradio():
    with gr.Blocks(title="逼真着装数字人生成系统设计与实现") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path = gr.Image(
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                    person_image = gr.ImageEditor(
                        interactive=True, label="人物图像", type="filepath"
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="服装图像", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">提供遮罩的两种方式：<br>1. 上传人物图像并使用上方的`🖌️`绘制遮罩（优先级更高）<br>2. 选择`试穿服装类型`自动生成遮罩</span>'
                        )
                        cloth_type = gr.Radio(
                            label="试穿服装类型",
                            choices=["上衣", "下装", "全身"],
                            value="上衣",
                        )


                submit = gr.Button("提交")
                gr.Markdown(
                    '<center><span style="color: #FF0000">!!! 请只点击一次，等待处理完成 !!!</span></center>'
                )

                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">高级选项可以调整细节：<br>1. `推理步数`可以增强细节；<br>2. `CFG强度`与饱和度高度相关；<br>3. `随机种子`可以改善伪阴影。</span>'
                )
                with gr.Accordion("高级选项", open=False):
                    num_inference_steps = gr.Slider(
                        label="推理步数", minimum=10, maximum=100, step=5, value=50
                    )
                    # Guidence Scale
                    guidance_scale = gr.Slider(
                        label="CFG强度", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                    )
                    # Random Seed
                    seed = gr.Slider(
                        label="随机种子", minimum=-1, maximum=10000, step=1, value=42
                    )
                    show_type = gr.Radio(
                        label="显示类型",
                        choices=["仅结果", "输入和结果", "输入、遮罩和结果"],
                        value="仅结果",
                    )

            with gr.Column(scale=2, min_width=500):
                # 创建一个新的行来并排放置图像和3D模型
                with gr.Row():
                    # 为生成结果图像创建一个列，设置其宽度比例为1
                    with gr.Column(scale=1):
                        result_image = gr.Image(interactive=True, label="生成结果")
                    # 为3D模型预览创建一个列，设置其宽度比例为2
                    with gr.Column(scale=2):
                        model_3d_viewer = gr.Model3D(
                            label="三维模型预览",
                            interactive=True, # 设为True允许用户旋转和缩放模型
                            height=500  # 增加高度使模型显示窗口更长
                        )
                
                # 示例部分保持在下方不变
                with gr.Row():
                    # Photo Examples
                    root_path = "resource/demo/example"
                    with gr.Column():
                        men_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "men", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "men"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="男性示例 ①",
                        )
                        women_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "women", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "women"))
                            ],
                            examples_per_page=4,
                            inputs=image_path,
                            label="女性示例 ②",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">如果需要生成三维人体模型，请选择全身照片。</span>'
                        )
                    with gr.Column():
                        condition_upper_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "upper", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "upper"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="上衣示例",
                        )
                        condition_overall_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "overall", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "overall"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="全身服装示例",
                        )
                        condition_person_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "person", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "person"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="参考人物服装示例",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*服装示例来自互联网。</span>'
                        )


            image_path.change(
                person_example_fn, inputs=image_path, outputs=person_image
            )

            # 修改submit.click事件的输出，增加3D模型查看器
            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                [result_image, model_3d_viewer], # 同时输出图像和3D模型
            )
    demo.queue().launch(share=True, show_error=True, server_port=7879)


if __name__ == "__main__":
    app_gradio()
