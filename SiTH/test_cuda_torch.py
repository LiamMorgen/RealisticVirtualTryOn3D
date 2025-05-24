import torch
import sys
import platform

def check_pytorch_cuda():
    # 系统信息
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        # CUDA版本
        print(f"CUDA版本: {torch.version.cuda}")
        
        # cuDNN版本
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"cuDNN是否可用: {cudnn_available}")
        if cudnn_available:
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        
        # GPU信息
        gpu_count = torch.cuda.device_count()
        print(f"可用GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
            print(f"  总内存: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    # 检查特定PyTorch功能
    has_float8 = hasattr(torch, 'float8_e4m3fn')
    print(f"\n是否支持float8_e4m3fn: {has_float8}")
    
    # 测试简单的CUDA张量操作
    if cuda_available:
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print("\nCUDA张量运算测试: 成功")
        except Exception as e:
            print(f"\nCUDA张量运算测试: 失败 - {str(e)}")

if __name__ == "__main__":
    check_pytorch_cuda()
