#!/usr/bin/env python
"""
测试损失函数是否正常工作
"""
import torch
from loss_factory import HybridLoss, HybridLossWithSmooth

def test_hybrid_loss():
    """测试原始HybridLoss"""
    print("Testing HybridLoss...")
    loss_func = HybridLoss(
        n_fft=128,
        hop_len=48,
        win_len=128,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30,
        lamda_mag=70
    )

    # 模拟输入
    batch_size = 4
    signal_length = 24000  # 1秒 @ 24kHz
    y_pred = torch.randn(batch_size, signal_length)
    y_true = torch.randn(batch_size, signal_length)

    loss = loss_func(y_pred, y_true)
    print(f"HybridLoss: {loss.item():.4f} ✓")
    return True

def test_hybrid_loss_with_smooth():
    """测试带平滑约束的HybridLoss"""
    print("\nTesting HybridLossWithSmooth...")

    # 测试不同smoothness_type
    for smoothness_type in ['l2', 'hps', 'tv']:
        print(f"  Testing smoothness_type='{smoothness_type}'...")
        loss_func = HybridLossWithSmooth(
            n_fft=128,
            hop_len=48,
            win_len=128,
            compress_factor=0.3,
            eps=1e-12,
            lamda_ri=30,
            lamda_mag=70,
            enable_smooth=True,
            smoothness_type=smoothness_type,
            weight_smooth_freq=0.15,
            weight_smooth_time=0.05
        )

        # 模拟输入
        batch_size = 2
        signal_length = 12000  # 0.5秒 @ 24kHz
        y_pred = torch.randn(batch_size, signal_length)
        y_true = torch.randn(batch_size, signal_length)

        loss = loss_func(y_pred, y_true)
        print(f"    {smoothness_type}: {loss.item():.4f} ✓")

    return True

def test_edge_cases():
    """测试边缘情况"""
    print("\nTesting edge cases...")

    # 1. 单样本
    loss_func = HybridLossWithSmooth(
        n_fft=128,
        hop_len=48,
        win_len=128,
        enable_smooth=True,
        smoothness_type='hps'
    )
    y_pred = torch.randn(1, 6000)
    y_true = torch.randn(1, 6000)
    loss = loss_func(y_pred, y_true)
    print(f"  Single sample: {loss.item():.4f} ✓")

    # 2. 大batch
    y_pred = torch.randn(16, 12000)
    y_true = torch.randn(16, 12000)
    loss = loss_func(y_pred, y_true)
    print(f"  Large batch (16): {loss.item():.4f} ✓")

    # 3. enable_smooth=False
    loss_func_no_smooth = HybridLossWithSmooth(
        n_fft=128,
        hop_len=48,
        win_len=128,
        enable_smooth=False
    )
    loss = loss_func_no_smooth(y_pred, y_true)
    print(f"  No smooth: {loss.item():.4f} ✓")

    return True

if __name__ == '__main__':
    print("="*60)
    print("Loss Function Test Suite")
    print("="*60)

    try:
        test_hybrid_loss()
        test_hybrid_loss_with_smooth()
        test_edge_cases()

        print("\n" + "="*60)
        print("All tests passed! ✓✓✓")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
