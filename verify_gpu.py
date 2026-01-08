#!/usr/bin/env python3
"""
GPU 功能验证脚本
验证 PyTorch 2.9.1 + RTX 5090 是否正常工作
"""

import sys

print("="*60)
print("  PyTorch GPU 验证")
print("="*60)

try:
    import torch
    print(f"\n✅ PyTorch 导入成功")
    print(f"   版本: {torch.__version__}")
except ImportError as e:
    print(f"\n❌ PyTorch 导入失败: {e}")
    sys.exit(1)

# 检查 CUDA 可用性
print(f"\n{'='*60}")
print("  CUDA 状态检查")
print("="*60)

cuda_available = torch.cuda.is_available()
print(f"\nCUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")

if not cuda_available:
    print("\n⚠️  CUDA 不可用，可能的原因：")
    print("  1. PyTorch 编译时未包含 CUDA 支持")
    print("  2. NVIDIA 驱动问题")
    print("  3. 环境变量配置问题")
    sys.exit(1)

# CUDA 详细信息
print(f"CUDA 版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"cuDNN 启用: {'✅ 是' if torch.backends.cudnn.enabled else '❌ 否'}")

# GPU 信息
print(f"\n{'='*60}")
print("  GPU 设备信息")
print("="*60)

gpu_count = torch.cuda.device_count()
print(f"\nGPU 数量: {gpu_count}")

for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}:")
    print(f"  名称: {torch.cuda.get_device_name(i)}")
    print(f"  计算能力: {props.major}.{props.minor}")
    print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
    print(f"  多处理器数量: {props.multi_processor_count}")

# 内存状态
print(f"\n{'='*60}")
print("  显存使用情况 (GPU 0)")
print("="*60)

print(f"\n已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"已保留: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
print(f"最大已分配: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

# 简单计算测试
print(f"\n{'='*60}")
print("  GPU 计算测试")
print("="*60)

try:
    print("\n测试 1: 张量创建和转移")
    x = torch.randn(1000, 1000).cuda()
    print("  ✅ 张量创建成功")

    print("\n测试 2: 矩阵乘法 (FP32)")
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print(f"  ✅ 矩阵乘法成功，结果形状: {z.shape}")

    print("\n测试 3: FP16 精度")
    x_fp16 = x.half()
    y_fp16 = y.half()
    z_fp16 = torch.matmul(x_fp16, y_fp16)
    torch.cuda.synchronize()
    print(f"  ✅ FP16 计算成功，结果形状: {z_fp16.shape}")

    print("\n测试 4: 释放显存")
    del x, y, z, x_fp16, y_fp16, z_fp16
    torch.cuda.empty_cache()
    print("  ✅ 显存释放成功")

    print(f"\n当前显存: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

except Exception as e:
    print(f"\n❌ GPU 计算测试失败: {e}")
    sys.exit(1)

# RTX 5090 特定检查
print(f"\n{'='*60}")
print("  RTX 5090 兼容性检查")
print("="*60)

props = torch.cuda.get_device_properties(0)
compute_cap = f"{props.major}.{props.minor}"
compute_cap_value = props.major * 10 + props.minor

print(f"\n计算能力: {compute_cap} (sm_{props.major}{props.minor})")

if compute_cap_value >= 120:  # sm_120 (RTX 5090)
    print("  ✅ RTX 5090 (sm_120) 完全支持！")
    print("  ✅ 可以使用所有现代 CUDA 特性")
elif compute_cap_value >= 90:
    print("  ✅ 现代 GPU，支持大部分特性")
elif compute_cap_value >= 80:
    print("  ⚠️  较旧的 GPU 架构")
else:
    print("  ⚠️  非常旧的 GPU 架构，可能性能受限")

# 最终总结
print(f"\n{'='*60}")
print("  验证总结")
print("="*60)

print("\n✅ 所有测试通过！")
print("\nPyTorch 2.9.1 已成功配置，支持：")
print(f"  • {gpu_count} × {torch.cuda.get_device_name(0)}")
print(f"  • 计算能力: {compute_cap}")
print(f"  • CUDA {torch.version.cuda}")
print(f"  • 总显存: {props.total_memory / 1024**3:.2f} GB × {gpu_count}")
print("\n🚀 现在可以启用 GPU 模式获得最佳性能！")

print("\n" + "="*60)
