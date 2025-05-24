import argparse
import json
import os
from PIL import Image
from dwpose import DwposeDetector

def convert_to_target_format(dwpose_json):
    # 创建目标格式的JSON
    target_json = {
        "version": 1.3,
        "people": []
    }
    
    # 处理每个人物数据
    if "people" in dwpose_json and dwpose_json["people"]:
        for person in dwpose_json["people"]:
            person_data = {
                "person_id": [-1],
            }
            
            # 复制所有关键点数据
            for key in person:
                person_data[key] = person[key]
            
            target_json["people"].append(person_data)
    
    return target_json

def process_single_image(image_path, model, output_json_path=None):
    """处理单张图片并生成关节点JSON"""
    # 检查输入文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 未找到输入图片 {image_path}")
        return

    if not output_json_path:
        # 如果未指定输出路径，则根据输入图像名生成
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        # 确保输出目录存在，如果输入路径是相对路径，则输出目录可能为空
        if not output_dir:
            output_dir = "."
        output_json_path = os.path.join(output_dir, f"{base_name}.json")

    print(f"处理图片: {image_path}")
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"打开图片错误 {image_path}: {e}")
        return

    # 运行模型
    try:
        imgOut, j, source = model(img,
                                 include_hand=True,
                                 include_face=True,
                                 include_body=True,
                                 image_and_json=True,
                                 detect_resolution=1024)
    except Exception as e:
        print(f"姿态估计过程中出错: {e}")
        return

    # 转换JSON格式
    converted_json = convert_to_target_format(j)

    # 保存转换后的 JSON 文件
    try:
        with open(output_json_path, "w") as f:
            f.write(json.dumps(converted_json, indent=4)) # 添加缩进使JSON更易读
        print(f"已保存转换后的JSON到: {output_json_path}")
    except Exception as e:
        print(f"保存JSON文件到 {output_json_path} 时出错: {e}")
        return

    # # 保存带有姿势估计的图像 (使用固定或衍生名称)
    # output_image_path = os.path.join(os.path.dirname(output_json_path), f"{os.path.splitext(os.path.basename(image_path))[0]}_DWpose_output.jpg")
    # try:
    #     imgOut.save(output_image_path)
    #     print(f"已保存输出图像到: {output_image_path}")
    # except Exception as e:
    #     print(f"保存输出图像到 {output_image_path} 时出错: {e}")
    
    return output_json_path

def is_image_file(filename):
    """检查文件是否为支持的图片格式"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def main(args):
    input_path = args.image
    output_json_path = args.output

    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入路径不存在 {input_path}")
        return

    # 初始化模型 (只初始化一次)
    try:
        model = DwposeDetector.from_pretrained_default()
    except Exception as e:
        print(f"初始化DwposeDetector模型出错: {e}")
        return

    # 处理输入路径 (文件夹或单个图片)
    if os.path.isdir(input_path):
        # 处理文件夹内的所有图片
        processed_count = 0
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path) and is_image_file(filename):
                process_single_image(file_path, model)
                processed_count += 1
        
        print(f"已处理文件夹中的 {processed_count} 张图片。")
    else:
        # 处理单张图片
        process_single_image(input_path, model, output_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在图片上运行DWpose并以转换后的JSON格式保存关键点。")
    parser.add_argument("-i", "--image", type=str, required=True, help="输入图片文件路径或包含图片的文件夹路径。")
    parser.add_argument("-o", "--output", type=str, help="输出JSON文件路径。默认为与图片相同目录下的<图像名字>.json。")

    args = parser.parse_args()
    main(args)

