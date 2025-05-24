#!/usr/bin/env python3
import os
import sys

def test_symlink():
    """测试软链接文件夹访问情况"""
    
    # 要测试的软链接路径
    symlink_path = "/home/dengjunli/data/dengjunli2/DLBounty/SiTH/data/From_CatVTON"
    
    # 测试1：检查软链接是否存在
    if not os.path.exists(symlink_path):
        print(f"❌ 软链接 {symlink_path} 不存在!")
        return
    
    print(f"✅ 软链接 {symlink_path} 存在")
    
    # 测试2：检查是否是软链接
    if not os.path.islink(symlink_path):
        print(f"❌ {symlink_path} 不是软链接!")
        return
    
    print(f"✅ {symlink_path} 是软链接")
    print(f"   指向: {os.readlink(symlink_path)}")
    
    # 测试3：检查目标是否是目录
    if not os.path.isdir(symlink_path):
        print(f"❌ 软链接指向的不是目录!")
        return
    
    print(f"✅ 软链接指向的是目录")
    
    # 测试4：列出子文件夹
    try:
        subdirs = [d for d in os.listdir(symlink_path) 
                   if os.path.isdir(os.path.join(symlink_path, d))]
        print(f"✅ 可以列出子文件夹: {', '.join(subdirs)}")
    except Exception as e:
        print(f"❌ 无法列出子文件夹: {str(e)}")
        return
    
    # 测试5：在每个子文件夹中创建测试文件
    print("\n测试在子文件夹中创建文件:")
    for subdir in subdirs:
        test_filepath = os.path.join(symlink_path, subdir, "test_file.txt")
        try:
            with open(test_filepath, 'w') as f:
                f.write("这是一个测试文件")
            print(f"✅ 成功在 {subdir} 中创建文件")
            
            # 清理：删除测试文件
            os.remove(test_filepath)
        except Exception as e:
            print(f"❌ 无法在 {subdir} 中创建文件: {str(e)}")

if __name__ == "__main__":
    print("开始测试软链接文件夹访问...")
    test_symlink()
    print("测试完成!")