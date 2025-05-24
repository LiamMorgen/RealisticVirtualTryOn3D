"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os
import numpy as np
import PIL.Image as Image
import argparse
import cv2
import io

def remove_background(img_path):
    """
    使用rembg库去除图像背景
    
    Args:
        img_path: 输入图像路径
        
    Returns:
        抠图后的RGBA格式图像
    """
    print(f"正在抠图: {img_path}")
    
    try:
        # 尝试导入rembg，如果失败则返回原始图像
        try:
            from rembg import remove
        except ImportError:
            print("警告: 未安装rembg库，请使用pip install rembg[cpu]安装")
            return Image.open(img_path).convert("RGBA")
            
        # 方法1: 先加载为PIL图像，再处理
        try:
            input_img = Image.open(img_path).convert("RGB")
            output_img = remove(input_img)
            print(f"使用PIL方法成功抠图: {img_path}")
            return output_img
        except Exception as e:
            print(f"PIL方法抠图失败: {e}")

        # 方法2: 直接读取文件内容
        try:
            with open(img_path, 'rb') as img_file:
                input_data = img_file.read()
                output_data = remove(input_data)
                output_img = Image.open(io.BytesIO(output_data)).convert("RGBA")
                print(f"使用文件读取方法成功抠图: {img_path}")
                return output_img
        except Exception as e:
            print(f"文件读取方法抠图失败: {e}")
            
        # 方法3: 使用原始方法（用于兼容性）
        try:
            f = np.fromfile(img_path)
            result = remove(f)
            img = Image.open(io.BytesIO(result)).convert("RGBA")
            print(f"使用原始方法成功抠图: {img_path}")
            return img
        except Exception as e:
            print(f"原始方法抠图失败: {e}")
            
        # 所有方法都失败，回退到简单的RGBA转换
        print(f"所有抠图方法均失败，返回原始图像")
        return Image.open(img_path).convert("RGBA")
    except Exception as e:
        print(f"抠图过程中发生未知错误: {e}")
        # 如果抠图完全失败，返回原始图像
        try:
            return Image.open(img_path).convert("RGBA")
        except:
            print(f"无法打开原始图像: {img_path}")
            # 如果原始图像也无法打开，创建一个空白图像
            return Image.new("RGBA", (1024, 1024), (255, 255, 255, 0))

def main(args):

    # Make sure the output path exists
    os.makedirs(args.output_path, exist_ok=True)

    # Get the list of images
    img_list = [os.path.join(args.input_path, x) for x in sorted(os.listdir(args.input_path)) if x.endswith(('.png'))]

    for img_path in img_list:

        # 加载图像
        try:
            if args.remove_bg:
                # 使用抠图功能
                img = remove_background(img_path)
            else:
                img = Image.open(img_path)
                # 检查图像模式，如果不是RGBA，则转换为RGBA
                if img.mode != 'RGBA':
                    print(f"注意: {img_path} 不是RGBA格式图像，正在转换为RGBA格式")
                    img = img.convert('RGBA')
        except Exception as e:
            print(f"错误: 无法处理图像 {img_path}，错误信息: {e}")
            continue

        canvas = Image.new('RGBA', (args.size, args.size))

        img_width, img_height = img.size
        target_height = int( args.size * args.ratio)

        resize_ratio = target_height / img_height
        target_width = int(img_width * resize_ratio)

        img_resized = img.resize((target_width, target_height))

        rgb = np.asarray(img_resized)[..., :3]
        mask = np.asarray(img_resized)[..., 3]

        # Erode the mask to remove the noisy borders
        new_mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)

        img_resized = Image.fromarray(np.concatenate([rgb, new_mask[..., None]], axis=-1))

        #place resized image on the center
        x_offset = int((args.size - target_width) // 2)
        y_offset = int((args.size - target_height) // 2)

        canvas.paste(img_resized, (x_offset, y_offset))


        canvas.save(os.path.join(args.output_path, os.path.basename(img_path)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a square images given RGBA images')

    parser.add_argument("-i", "--input-path", default='./data/examples/rgba', type=str, help="Input RGBA path")
    parser.add_argument("-o", "--output-path", default='./data/examples/images', type=str, help="Output path")
    parser.add_argument("--ratio", default=0.85, type=float, help="Ratio of the image height to the canvas height")
    parser.add_argument("--size", default=1024, type=int, help="Canvas size")
    parser.add_argument("--remove-bg", action='store_true', help="使用rembg库去除图像背景")
    print("处理中... 图像位于 ", parser.parse_args().input_path)
    print(parser.parse_args())
    main(parser.parse_args())

