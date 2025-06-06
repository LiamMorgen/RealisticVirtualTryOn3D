
# 逼真着装数字人生成系统安装部署使用文档

## 系统简介

本系统由CatVTON和SiTH两个模块组成，实现了服装虚拟试穿及3D数字人模型生成功能。

## 环境要求

- CUDA 11.8或更高版本
- Python 3.8+
- 足够的GPU内存（建议≥24GB 双卡）

## 安装步骤

注意：
安装kaolin支持的torch版本（目前torch2.7不支持），说明：https://github.com/NVIDIAGameWorks/kaolin/
pip install pydantic==2.10.6
export HF_ENDPOINT=https://hf-mirror.com

### 1. 克隆代码仓库

```bash
git clone [仓库地址] DLBounty
cd DLBounty
```

### 2. 安装SiTH依赖
参考`SiTH/README.md`
来自：https://github.com/SiTH-Diffusion/SiTH

```bash
cd SiTH
pip install -r requirements.txt
```

### 3. 安装CatVTON依赖
参考`CatVTON/README.md`
来自：https://github.com/Zheng-Chong/CatVTON

```bash
cd ../CatVTON
pip install -r requirements_without_version.txt
```

### 4. 下载预训练模型

系统首次运行时会自动从HuggingFace下载必要的预训练模型。如无法访问HuggingFace，请手动下载模型并放置在相应目录。

## 使用方法

### 启动系统

```bash
cd CatVTON
python app.py
```

系统将启动Gradio界面，默认端口为7879，可通过浏览器访问：http://localhost:7879

### 使用流程

1. 上传或选择人物图像
2. 上传或选择服装图像
3. 选择试穿服装类型（上衣/下装/全身）
4. 点击"提交"按钮
5. 等待系统处理（生成虚拟试穿结果和3D模型，约需50秒）
6. 查看生成结果和3D模型

### 高级选项

- **推理步数**：增加可提高细节质量，但会延长处理时间
- **CFG强度**：影响图像饱和度和真实感
- **随机种子**：固定值可复现结果，-1为随机种子
- **显示类型**：可选择显示模式（仅结果/输入和结果/输入、遮罩和结果）

## 注意事项

1. 点击"提交"按钮后请耐心等待，勿重复点击
2. 生成3D模型需使用全身照片
3. 两个项目目录必须保持正确的相对位置关系
4. 系统会在首次运行时下载模型，请确保网络连接正常

## 常见问题

1. **模型下载失败**：检查网络连接或使用代理
2. **CUDA错误**：确认CUDA版本兼容性和GPU驱动更新
3. **内存不足**：减小图像尺寸或降低推理步数
4. **3D模型生成失败**：检查SiTH目录是否正确，确保提供了全身照片

## 系统结构

- **CatVTON**：负责虚拟试穿图像生成
- **SiTH**：负责3D数字人模型生成

若有其他问题，请参考项目README文件或提交issue。
