import time
import sys

import paddle
paddle.enable_compat()
import torch
import numpy as np
import kernelkit as kk

from lib import RawTestParamForPrefill, TestParam
import lib
import ref

_counter = kk.Counter()

@paddle.no_grad()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase_for_prefill(p)
    torch.cuda.synchronize()

    def run_prefill():
        return lib.run_flash_mla_sparse_fwd(p, t, False)

    prefill_ans_out, prefill_ans_max_logits, prefill_ans_lse = run_prefill()
    torch.cuda.synchronize()

    if p.save_results:
        np.savez(f'ans/prefill_b{p.prefill.b}s{p.prefill.seqlens[0]}.npz',
                 prefill_ans_out=prefill_ans_out.detach().cpu().float().numpy(),
                 prefill_ans_max_logits=prefill_ans_max_logits.detach().cpu().numpy(),
                 prefill_ans_lse=prefill_ans_lse.detach().cpu().numpy())

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol(p, t)
        prefill_ans_time = kk.bench_paddle(run_prefill, num_tests=p.num_runs).get_kernel_time("sparse_attn_fwd")
        prefill_flops = flops_and_mem_vol.fwd_flop/prefill_ans_time/1e12
        prefill_mem_bw = flops_and_mem_vol.fwd_mem_vol/prefill_ans_time/1e12
        print(f"Prefill:  {prefill_ans_time*1e6:4.0f} us, {prefill_flops:6.1f} TFlops, {prefill_mem_bw:4.2f} TBps")

    if p.check_correctness:
        torch.cuda.synchronize()
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        torch.cuda.synchronize()

        is_correct = True
        is_correct &= kk.check_is_allclose("out", prefill_ans_out.float(), ref_out_fp32, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        is_correct &= kk.check_is_allclose("max_logits", prefill_ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        is_correct &= kk.check_is_allclose("lse", prefill_ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)

        return is_correct
    else:
        return True


if __name__ == '__main__':
    device = paddle.device("cuda:2")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    # torch.set_float32_matmul_precision('high')

    performance_case_templates = {
        "model_configs": [
            # V3.2
            (576, 128, 512, 1, 2048),   # d_qk, h_q, d_v, h_kv, topk
            (576, 128, 512, 1, 4096),
            # MODEL1 CONFIG1
            (512, 64, 512, 1, 512),
            # MODEL1 CONFIG2
            (512, 128, 512, 1, 1024),
        ],
        "seqlens": [2048, 4096, 8192, 16384, 32768, 65536, 131072],
        "batch_sizes": [1, 2, 3, 4, 5, 10, 20, 40, 60]
    }

    performance_cases = [
        RawTestParamForPrefill(
            bsz, bsz * [seqlens], topk, h_q=h_q, d_qk=d_qk, h_kv=h_kv, d_v=d_v,
            have_attn_sink=True, check_correctness=False).to_test_param()
        for (d_qk, h_q, d_v, h_kv, topk) in performance_case_templates["model_configs"]
        for seqlens in performance_case_templates["seqlens"]
        for bsz in performance_case_templates["batch_sizes"]
    ]

    testcases = performance_cases

    is_no_cooldown = lib.is_no_cooldown()
    failed_cases = []
    for test in testcases:
        # lib.generate_testcase_for_prefill(test).dump_as_npz(f"./cases/prefill_b{test.prefill.b}s{test.prefill.seqlens[0]}.npz")
        # comment below if you only want to save testcases
        if test != testcases[0] and test.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        is_correct = run_test(test)
        if is_correct:
            print("\033[32m\033[1mPASSED\033[0m")
        else:
            print("\033[31m\033[1mFAILED\033[0m")
            failed_cases.append(test)

    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")
