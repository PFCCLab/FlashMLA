import os

import paddle
paddle.enable_compat()
import torch
import numpy as np
import sys
sys.path.append('./tests')
import kernelkit as kk

def check_precision(ans_out: torch.Tensor, ans_lse: torch.Tensor,
                    ref_out: torch.Tensor, ref_lse: torch.Tensor) -> bool:
    """
    对比 Paddle 和 Torch 的执行结果
    
    数据加载说明:
    - out_ans: numpy.float32 -> 对比时直接使用 float32
    - lse_ans: numpy.float32 -> 对比时直接使用 float32
    
    注意: 保存时已从 bfloat16 转为 float32，对比在 float32 精度下进行
    """
    is_correct = True
    is_correct &= kk.check_is_allclose("out", ans_out, ref_out, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
    is_correct &= kk.check_is_allclose("lse", ans_lse, ref_lse, abs_tol=1e-6, rel_tol=8.01/65536)
    return is_correct

# 配置
enable_precision_check = True
base_dir = "../../compare"
paddle_dir = os.path.join(base_dir, "output/paddle")
torch_dir = os.path.join(base_dir, "output/torch")

# 测试参数 - 根据实际生成的测试用例调整 (b, s, h_q, d_qk, topk)
test_params = [
    (4, 16384, 128, 576, 2048),
    (2, 32768, 128, 576, 2048),
    (1, 65536, 128, 576, 2048),
    (4, 16384, 128, 576, 4096),
    (2, 32768, 128, 576, 4096),
    (1, 65536, 128, 576, 4096),
    (148, 32768, 64, 512, 16384),
    (148, 32768, 64, 576, 16384),
    (148, 32768, 128, 512, 16384),
    (148, 32768, 128, 576, 16384),
]

if __name__ == '__main__':
    device = paddle.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    passed = 0
    failed = 0
    
    for b, s, h_q, d_qk, topk in test_params:
        print("================")
        print(f"Checking precision on b={b}, s={s}, h_q={h_q}, d_qk={d_qk}, topk={topk}")

        result_file = f"decoding_sparse_b{b}_s{s}_h{h_q}_d{d_qk}_topk{topk}.npz"
        paddle_path = os.path.join(paddle_dir, result_file)
        torch_path = os.path.join(torch_dir, result_file)
        
        # 检查文件是否存在
        if not os.path.exists(paddle_path):
            print(f"  Paddle result not found: {paddle_path}")
            continue
        if not os.path.exists(torch_path):
            print(f"  Torch result not found: {torch_path}")
            continue

        # 加载 Paddle 结果 (作为 ans)
        # 数据类型: numpy.float32 (保存时已从 bfloat16 转换)
        with np.load(paddle_path) as paddle_data:
            ans_out = torch.tensor(paddle_data["out_ans"], dtype=torch.float32)
            ans_lse = torch.tensor(paddle_data["lse_ans"], dtype=torch.float32)

        # 加载 Torch 结果 (作为 ref)
        # 数据类型: numpy.float32 (保存时已从 bfloat16 转换)
        with np.load(torch_path) as torch_data:
            ref_out = torch.tensor(torch_data["out_ans"], dtype=torch.float32)
            ref_lse = torch.tensor(torch_data["lse_ans"], dtype=torch.float32)

        # 计算差异
        diff_out = ans_out - ref_out
        diff_lse = ans_lse - ref_lse

        print(f"  out shape: {ans_out.shape}")
        print(f"  lse shape: {ans_lse.shape}")
        print(f"  diff_out sum: {diff_out.sum().item():.6e}")
        print(f"  diff_out abs max: {diff_out.abs().max().item():.6e}")
        print(f"  diff_lse sum: {diff_lse.sum().item():.6e}")
        print(f"  diff_lse abs max: {diff_lse.abs().max().item():.6e}")

        if enable_precision_check:
            if check_precision(ans_out, ans_lse, ref_out, ref_lse):
                print(f"  \033[32m\033[1mPASSED\033[0m")
                passed += 1
            else:
                print(f"  \033[31m\033[1mFAILED\033[0m")
                failed += 1
    
    print("================")
    print(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")