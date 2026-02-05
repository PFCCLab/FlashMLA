import os

import paddle
paddle.enable_compat()
import torch
import numpy as np
import kernelkit as kk

def check_precision(ans_out: torch.Tensor, ans_max_logits: torch.Tensor, ans_lse: torch.Tensor,
                    ref_out: torch.Tensor, ref_max_logits: torch.Tensor, ref_lse: torch.Tensor) -> bool:
    is_correct = True
    is_correct &= kk.check_is_allclose("out", ans_out, ref_out, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
    is_correct &= kk.check_is_allclose("max_logits", ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
    is_correct &= kk.check_is_allclose("lse", ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)
    return is_correct

# may cause cuda: out of memory error
enable_precision_check = False
ans_dir = "./ans"
ref_dir = "./ref"
batch_sizes = [4]
sequence_lengths = [8192]

# only tested under pytorch
if __name__ == '__main__':
  device = paddle.device("cuda:0")
  torch.set_default_device(device)
  torch.cuda.set_device(device)

  for b, s in zip(batch_sizes, sequence_lengths):
    print("================")
    print(f"Checking precision on batch_size: {b}, sequence_length: {s}")

    result_file = f"prefill_b{b}s{s}.npz"

    # load answer data
    with np.load(os.path.abspath(os.path.join(ans_dir, result_file))) as ans_data:
      ans_out = torch.tensor(ans_data["prefill_ans_out"], dtype=torch.float32)
      ans_max_logits = torch.tensor(ans_data["prefill_ans_max_logits"], dtype=torch.float32)
      ans_lse = torch.tensor(ans_data["prefill_ans_lse"], dtype=torch.float32)

    # load reference data
    with np.load(os.path.abspath(os.path.join(ref_dir, result_file))) as ref_data:
      ref_out = torch.tensor(ref_data["prefill_ans_out"], dtype=torch.float32)
      ref_max_logits = torch.tensor(ref_data["prefill_ans_max_logits"], dtype=torch.float32)
      ref_lse = torch.tensor(ref_data["prefill_ans_lse"], dtype=torch.float32)

    diff_out = ans_out - ref_out
    diff_max_logits = ans_max_logits - ref_max_logits
    diff_lse = ans_lse - ref_lse

    print(f"diff_out: {diff_out.sum().item()}")
    print(f"diff_max_logits: {diff_max_logits.sum().item()}")
    print(f"diff_lse: {diff_lse.sum().item()}")

    if enable_precision_check:
      if check_precision(ans_out, ans_max_logits, ans_lse, ref_out, ref_max_logits, ref_lse):
          print(f"\033[32m\033[1mPASSED\033[0m")
      else:
          print(f"\033[31m\033[1mFAILED\033[0m")
